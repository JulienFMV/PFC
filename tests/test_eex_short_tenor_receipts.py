from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from pfc_shaping.pipeline import production_phases
from pfc_shaping.validation.eex_short_tenor_receipts import (
    COMBINATION_PROOF_CONTENT_ID,
    COMBINATION_PROOF_MANIFEST_SHA256,
    CONTRACT_CONTENT_ID,
    CONTRACT_STATUS,
    CONTRAST_BUNDLE_CONTENT_ID,
    CONTRAST_BUNDLE_MANIFEST_SHA256,
    EVIDENCE_SELECTION_CONTENT_ID,
    EVIDENCE_SELECTION_DOCUMENT_SHA256,
    EVIDENCE_SELECTION_PROOF_CONTENT_ID,
    EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256,
    POLICY_CONTENT_ID,
    POLICY_DOCUMENT_SHA256,
    RECEIPT_STATUS,
    REQUIRED_MATERIAL_ROLES,
    SELECTION_RECEIPT_SCHEMA,
    SELECTION_RULE,
    SHAPE_PROJECTION_CONTENT_ID,
    SUCCESSOR_CORE_ID,
    TRAINING_RECEIPT_SCHEMA,
    ShortTenorReceiptError,
    load_strict_receipt_json,
    validate_short_tenor_receipt_contract,
    validate_structural_selection_receipt,
    validate_structural_training_receipt,
    with_receipt_id,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "EEX-CH-SHORT-TENOR-RECEIPT-CONTRACT-V1.json"
)


def _h(character: str) -> str:
    return character * 64


def _json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _access_log() -> dict[str, bool]:
    return {
        "outer_target_truth_opened": False,
        "ompex_opened": False,
        "afry_numeric_values_accessed": False,
        "t057_accessed": False,
        "future_holdout_consumed": False,
        "post_origin_data_accessed": False,
    }


def _claims() -> dict[str, bool]:
    return {
        "receipt_authenticated": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
    }


def _authority() -> dict[str, object]:
    return {
        "governance_key_id": _h("1"),
        "execution_key_id": _h("2"),
        "trusted_time_key_id": _h("3"),
        "source_acquisition_key_ids": [_h("4"), _h("5")],
        "source_trusted_time_key_ids": [_h("6"), _h("7")],
        "cryptographic_authentication_performed": False,
        "independent_authorities_proven": False,
    }


def _materials() -> list[dict[str, str]]:
    content_ids = {
        "EEX_PIT_CATALOG": _h("1"),
        "ENTSOE_PIT_CATALOG": _h("2"),
        "D225_SELECTION_DOCUMENT": EVIDENCE_SELECTION_CONTENT_ID,
        "D225_SELECTION_PROOF": EVIDENCE_SELECTION_PROOF_CONTENT_ID,
        "D220_CONTRAST_BUNDLE": CONTRAST_BUNDLE_CONTENT_ID,
        "D225_COMBINATION_PROOF": COMBINATION_PROOF_CONTENT_ID,
        "D219_PROJECTION_PROOF": SHAPE_PROJECTION_CONTENT_ID,
        "D225_COMBINATION_POLICY": POLICY_CONTENT_ID,
        "ORIGIN_TARGET_MASK": _h("6"),
        "CANDIDATE_GRID": _h("7"),
        "FOLD_INVENTORY": _h("8"),
        "IMPLEMENTATION_CLOSURE": _h("9"),
        "RUNTIME_LOCK": _h("a"),
    }
    digests = {
        role: _h(character)
        for role, character in zip(
            REQUIRED_MATERIAL_ROLES, "123456789abcd", strict=True
        )
    }
    digests["D225_SELECTION_DOCUMENT"] = EVIDENCE_SELECTION_DOCUMENT_SHA256
    digests["D225_SELECTION_PROOF"] = EVIDENCE_SELECTION_PROOF_MANIFEST_SHA256
    digests["D220_CONTRAST_BUNDLE"] = CONTRAST_BUNDLE_MANIFEST_SHA256
    digests["D225_COMBINATION_PROOF"] = COMBINATION_PROOF_MANIFEST_SHA256
    digests["D225_COMBINATION_POLICY"] = POLICY_DOCUMENT_SHA256
    digests["CANDIDATE_GRID"] = _h("b")
    digests["FOLD_INVENTORY"] = _h("c")
    return [
        {
            "role": role,
            "name": role.lower(),
            "sha256": digests[role],
            "content_id": content_ids[role],
            "max_available_at_utc": "2027-07-31T00:00:00Z",
            "trusted_time_receipt_sha256": _h("f"),
        }
        for role in REQUIRED_MATERIAL_ROLES
    ]


def _inner_folds() -> list[dict[str, object]]:
    return [
        {
            "fold_id": "F01",
            "fold_as_of_utc": "2026-12-01T00:00:00Z",
            "training_origin_ids": ["O01"],
            "validation_origin_id": "V01",
            "training_data_max_available_at_utc": "2026-11-30T23:00:00Z",
            "validation_target_delivery_start_utc": "2027-01-01T00:00:00Z",
            "validation_target_delivery_end_utc": "2027-02-01T00:00:00Z",
            "validation_truth_available_at_utc": "2027-02-02T00:00:00Z",
            "fold_inventory_entry_sha256": _h("a"),
            "eex_pit_cutoff_receipt_sha256": _h("b"),
            "entsoe_pit_cutoff_receipt_sha256": _h("c"),
            "fold_input_snapshot_sha256": _h("d"),
        },
        {
            "fold_id": "F02",
            "fold_as_of_utc": "2027-05-01T00:00:00Z",
            "training_origin_ids": ["O01", "V01"],
            "validation_origin_id": "V02",
            "training_data_max_available_at_utc": "2027-04-30T23:00:00Z",
            "validation_target_delivery_start_utc": "2027-06-01T00:00:00Z",
            "validation_target_delivery_end_utc": "2027-07-01T00:00:00Z",
            "validation_truth_available_at_utc": "2027-07-02T00:00:00Z",
            "fold_inventory_entry_sha256": _h("b"),
            "eex_pit_cutoff_receipt_sha256": _h("c"),
            "entsoe_pit_cutoff_receipt_sha256": _h("d"),
            "fold_input_snapshot_sha256": _h("e"),
        },
    ]


def _run_details(*, selection: bool, access: dict[str, bool]) -> dict[str, object]:
    byproducts = {
        "access_log": _json_hash(access),
        "environment_manifest": _h("8"),
        "selection_log" if selection else "training_log": _h("9"),
    }
    if not selection:
        byproducts["inner_loss_commitment"] = _h("e")
    return {
        "builder_id": "https://fmv.example.invalid/builders/short-tenor-fixture-v1",
        "invocation_id": "selection-fixture-001" if selection else "training-fixture-001",
        "started_at_utc": (
            "2027-09-01T00:00:00Z" if selection else "2027-08-01T00:00:00Z"
        ),
        "finished_at_utc": (
            "2027-10-01T00:00:00Z" if selection else "2027-09-01T00:00:00Z"
        ),
        "execution_attestation_sha256": _h("a"),
        "trusted_time_receipt_sha256": _h("b"),
        "byproduct_sha256s": byproducts,
    }


def _training_receipt() -> dict[str, object]:
    access = _access_log()
    unsigned: dict[str, object] = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "receipt_role": "TRAINING",
        "status": RECEIPT_STATUS,
        "market": "CH",
        "currency_unit": "EUR/MWh",
        "campaign_type": "SELECTION",
        "outer_origin_id": "OUTER-2028-01",
        "outer_origin_as_of_utc": "2028-01-01T00:00:00Z",
        "outer_origin_entry_sha256": _h("0"),
        "plan_frozen_at_utc": "2026-01-01T00:00:00Z",
        "policy_content_id": POLICY_CONTENT_ID,
        "successor_core_id": SUCCESSOR_CORE_ID,
        "candidate_grid_sha256": _h("b"),
        "fold_inventory_sha256": _h("c"),
        "inner_loss_commitment_sha256": _h("e"),
        "materials": _materials(),
        "candidate_config_sha256_allowlist": [_h("d"), _h("e")],
        "inner_folds": _inner_folds(),
        "subject": {
            "role": "TRAINED_CANDIDATE_SET",
            "name": "trained-candidate-set.json",
            "sha256": _h("f"),
        },
        "run_details": _run_details(selection=False, access=access),
        "access_log": access,
        "authority": _authority(),
        "claims": _claims(),
    }
    return with_receipt_id(unsigned)


def _selection_receipt(training: dict[str, object]) -> dict[str, object]:
    access = _access_log()
    unsigned: dict[str, object] = {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "receipt_role": "SELECTION",
        "status": RECEIPT_STATUS,
        "market": "CH",
        "currency_unit": "EUR/MWh",
        "campaign_type": "SELECTION",
        "outer_origin_id": training["outer_origin_id"],
        "outer_origin_as_of_utc": training["outer_origin_as_of_utc"],
        "outer_origin_entry_sha256": training["outer_origin_entry_sha256"],
        "policy_content_id": POLICY_CONTENT_ID,
        "successor_core_id": SUCCESSOR_CORE_ID,
        "training_receipt_id": training["receipt_id"],
        "training_receipt_sha256": _json_hash(training),
        "candidate_grid_sha256": training["candidate_grid_sha256"],
        "fold_inventory_sha256": training["fold_inventory_sha256"],
        "selected_candidate_config_sha256": _h("d"),
        "selection_rule": SELECTION_RULE,
        "inner_loss_commitment_sha256": training[
            "inner_loss_commitment_sha256"
        ],
        "subject": {
            "role": "SELECTED_SHORT_TENOR_CANDIDATE",
            "name": "selected-short-tenor-candidate.json",
            "sha256": _h("f"),
            "candidate_config_sha256": _h("d"),
        },
        "run_details": _run_details(selection=True, access=access),
        "access_log": access,
        "authority": _authority(),
        "claims": _claims(),
    }
    return with_receipt_id(unsigned)


def _resign(receipt: dict[str, object]) -> None:
    receipt.pop("receipt_id", None)
    receipt.update(with_receipt_id(receipt))


def test_contract_is_structural_and_non_authoritative() -> None:
    result = validate_short_tenor_receipt_contract(_contract())

    assert result["status"] == CONTRACT_STATUS
    assert result["evidence_selection_content_id"] == EVIDENCE_SELECTION_CONTENT_ID
    assert result["evidence_selection_proof_content_id"] == (
        EVIDENCE_SELECTION_PROOF_CONTENT_ID
    )
    assert result["policy_content_id"] == POLICY_CONTENT_ID
    assert result["combination_proof_content_id"] == COMBINATION_PROOF_CONTENT_ID
    assert result["required_material_roles"] == list(REQUIRED_MATERIAL_ROLES)
    assert result["blocker_count"] == 8
    assert result["cryptographic_authentication_performed"] is False
    assert result["training_authorized"] is False
    assert result["selection_authorized"] is False
    assert result["production_authorized"] is False
    assert result["contract_content_id"] == CONTRACT_CONTENT_ID


def test_contract_rejects_stale_d225_or_hidden_nested_semantic_drift() -> None:
    changed_purpose = _contract()
    changed_purpose["purpose"] = "structural receipt contract with changed semantics"
    with pytest.raises(ShortTenorReceiptError, match="receipt contract content id"):
        validate_short_tenor_receipt_contract(changed_purpose)

    stale = _contract()
    stale["selected_boundary"]["evidence_selection_content_id"] = _h("0")
    with pytest.raises(ShortTenorReceiptError, match="selected boundary"):
        validate_short_tenor_receipt_contract(stale)

    unknown = _contract()
    unknown["common_receipt_contract"]["silent_override"] = True
    with pytest.raises(ShortTenorReceiptError, match="common receipt contract keys"):
        validate_short_tenor_receipt_contract(unknown)

    weakened = _contract()
    weakened["training_receipt_contract"][
        "training_finishes_not_after_outer_origin"
    ] = False
    with pytest.raises(ShortTenorReceiptError, match="training contract"):
        validate_short_tenor_receipt_contract(weakened)

    reordered = _contract()
    reordered["forbidden_inputs_and_claims"] = list(
        reversed(reordered["forbidden_inputs_and_claims"])
    )
    with pytest.raises(ShortTenorReceiptError, match="forbidden inputs and claims"):
        validate_short_tenor_receipt_contract(reordered)

    stale_profile = _contract()
    stale_profile["attestation_model"][
        "in_toto_attestation_framework_profile"
    ] = "v1.0"
    with pytest.raises(ShortTenorReceiptError, match="in-toto framework profile"):
        validate_short_tenor_receipt_contract(stale_profile)

    unknown_attestation = _contract()
    unknown_attestation["attestation_model"]["silent_override"] = True
    with pytest.raises(ShortTenorReceiptError, match="attestation model keys"):
        validate_short_tenor_receipt_contract(unknown_attestation)


def test_training_and_selection_receipts_link_without_granting_authority() -> None:
    training_receipt = _training_receipt()
    training = validate_structural_training_receipt(training_receipt)
    selection_receipt = _selection_receipt(training_receipt)
    selection = validate_structural_selection_receipt(
        selection_receipt,
        training_receipt=training_receipt,
    )

    assert training["inner_fold_ids"] == ["F01", "F02"]
    assert training["cryptographic_authentication_performed"] is False
    assert training["point_in_time_availability_proven"] is False
    assert training["training_authorized"] is False
    assert training["outer_origin_entry_sha256"] == _h("0")
    assert training["inner_loss_commitment_sha256"] == _h("e")
    assert selection["training_receipt_id"] == training["receipt_id"]
    assert selection["outer_origin_entry_sha256"] == training[
        "outer_origin_entry_sha256"
    ]
    assert selection["inner_loss_commitment_sha256"] == training[
        "inner_loss_commitment_sha256"
    ]
    assert selection["selected_candidate_config_sha256"] == _h("d")
    assert selection["cryptographic_authentication_performed"] is False
    assert selection["selection_authorized"] is False
    assert selection["candidate_assembly_authorized"] is False
    assert selection["production_authorized"] is False


def test_receipt_id_is_canonical_and_key_order_independent() -> None:
    training = _training_receipt()
    reversed_unsigned = dict(reversed([(k, v) for k, v in training.items() if k != "receipt_id"]))
    rebuilt = with_receipt_id(reversed_unsigned)

    assert rebuilt["receipt_id"] == training["receipt_id"]
    validate_structural_training_receipt(rebuilt)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("outer_origin_id", 1, "forbidden numeric"),
        ("outer_origin_as_of_utc", "2028-01-01T00:00:00+00:00", "canonical UTC"),
        ("policy_content_id", _h("0"), "policy content id"),
        ("successor_core_id", _h("0"), "successor core id"),
        ("candidate_grid_sha256", "BAD", "candidate grid"),
    ],
)
def test_training_rejects_header_numeric_time_policy_or_hash_drift(
    field: str, value: object, message: str
) -> None:
    receipt = _training_receipt()
    receipt[field] = value
    _resign(receipt)

    with pytest.raises(ShortTenorReceiptError, match=message):
        validate_structural_training_receipt(receipt)


def test_training_rejects_unknown_fields_and_bad_receipt_id() -> None:
    receipt = _training_receipt()
    receipt["unexpected"] = "field"
    with pytest.raises(ShortTenorReceiptError, match="keys differ"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["receipt_id"] = _h("0")
    with pytest.raises(ShortTenorReceiptError, match="canonical unsigned receipt id"):
        validate_structural_training_receipt(receipt)


def test_strict_receipt_loader_rejects_duplicate_keys_and_unbounded_bytes() -> None:
    duplicate = b'{"schema_version":"first","schema_version":"second"}'
    with pytest.raises(ShortTenorReceiptError, match="strict UTF-8 JSON"):
        load_strict_receipt_json(duplicate)

    with pytest.raises(ShortTenorReceiptError, match="bounded bytes"):
        load_strict_receipt_json(b"x" * (256 * 1024 + 1))


@pytest.mark.parametrize(
    ("role", "field", "value", "message"),
    [
        (
            "D225_SELECTION_DOCUMENT",
            "content_id",
            _h("0"),
            "D225 selection document content",
        ),
        (
            "D225_SELECTION_DOCUMENT",
            "sha256",
            _h("0"),
            "D225 selection document hash",
        ),
        (
            "D225_SELECTION_PROOF",
            "content_id",
            _h("0"),
            "D225 selection proof content",
        ),
        (
            "D225_SELECTION_PROOF",
            "sha256",
            _h("0"),
            "D225 selection proof manifest",
        ),
        ("D220_CONTRAST_BUNDLE", "content_id", _h("0"), "D220 material content"),
        ("D220_CONTRAST_BUNDLE", "sha256", _h("0"), "D220 material manifest"),
        (
            "D225_COMBINATION_PROOF",
            "content_id",
            _h("0"),
            "D225 combination proof content",
        ),
        (
            "D225_COMBINATION_PROOF",
            "sha256",
            _h("0"),
            "D225 combination proof manifest",
        ),
        (
            "D225_COMBINATION_POLICY",
            "content_id",
            _h("0"),
            "D225 policy material content",
        ),
        (
            "D225_COMBINATION_POLICY",
            "sha256",
            _h("0"),
            "D225 policy material document",
        ),
        ("CANDIDATE_GRID", "sha256", _h("0"), "grid material hash"),
        ("FOLD_INVENTORY", "sha256", _h("0"), "fold inventory material hash"),
        (
            "EEX_PIT_CATALOG",
            "max_available_at_utc",
            "2028-01-01T00:00:01Z",
            "available after outer origin",
        ),
    ],
)
def test_training_rejects_stale_unbound_or_post_origin_materials(
    role: str, field: str, value: object, message: str
) -> None:
    receipt = _training_receipt()
    material = next(item for item in receipt["materials"] if item["role"] == role)
    material[field] = value
    _resign(receipt)

    with pytest.raises(ShortTenorReceiptError, match=message):
        validate_structural_training_receipt(receipt)


def test_training_rejects_material_order_name_or_content_id_ambiguity() -> None:
    receipt = _training_receipt()
    receipt["materials"][0], receipt["materials"][1] = (
        receipt["materials"][1],
        receipt["materials"][0],
    )
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="material role order"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["materials"][1]["name"] = receipt["materials"][0]["name"]
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="names must be unique"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["materials"][1]["content_id"] = receipt["materials"][0]["content_id"]
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="content IDs must be unique"):
        validate_structural_training_receipt(receipt)


def test_training_rejects_unsafe_material_or_subject_logical_names() -> None:
    receipt = _training_receipt()
    receipt["materials"][0]["name"] = "../eex-catalog.json"
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="safe relative POSIX"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["materials"][0]["name"] = "catalogs\\eex.json"
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="safe relative POSIX"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["subject"]["name"] = "C:/selected.json"
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="safe relative POSIX"):
        validate_structural_training_receipt(receipt)


@pytest.mark.parametrize(
    ("fold_index", "field", "value", "message"),
    [
        (0, "training_data_max_available_at_utc", "2026-12-02T00:00:00Z", "PIT"),
        (0, "fold_as_of_utc", "2027-01-02T00:00:00Z", "PIT"),
        (0, "validation_truth_available_at_utc", "2028-01-02T00:00:00Z", "exceeds"),
        (0, "validation_origin_id", "O01", "appears in fold training"),
        (0, "validation_origin_id", "V02", "must be unique"),
        (
            0,
            "validation_origin_id",
            "OUTER-2028-01",
            "outer origin appears as inner-fold validation",
        ),
        (
            0,
            "eex_pit_cutoff_receipt_sha256",
            "BAD",
            "EEX PIT cutoff receipt",
        ),
        (
            0,
            "entsoe_pit_cutoff_receipt_sha256",
            "BAD",
            "ENTSO-E PIT cutoff receipt",
        ),
        (0, "fold_inventory_entry_sha256", "BAD", "fold inventory entry"),
        (0, "fold_input_snapshot_sha256", "BAD", "fold input snapshot"),
    ],
)
def test_training_rejects_inner_fold_leakage_or_time_travel(
    fold_index: int, field: str, value: object, message: str
) -> None:
    receipt = _training_receipt()
    receipt["inner_folds"][fold_index][field] = value
    _resign(receipt)

    with pytest.raises(ShortTenorReceiptError, match=message):
        validate_structural_training_receipt(receipt)


def test_training_rejects_unsorted_folds_origins_or_allowlist() -> None:
    receipt = _training_receipt()
    receipt["inner_folds"].reverse()
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="fold IDs must be sorted"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["inner_folds"][1]["training_origin_ids"] = ["V01", "O01"]
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="training origin IDs"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["candidate_config_sha256_allowlist"].reverse()
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="allowlist must be sorted"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["inner_folds"][0]["training_origin_ids"] = ["OUTER-2028-01"]
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="outer origin appears"):
        validate_structural_training_receipt(receipt)


@pytest.mark.parametrize("field", list(_access_log()))
def test_training_rejects_any_outcome_or_forbidden_source_access(field: str) -> None:
    receipt = _training_receipt()
    receipt["access_log"][field] = True
    receipt["run_details"]["byproduct_sha256s"]["access_log"] = _json_hash(
        receipt["access_log"]
    )
    _resign(receipt)

    with pytest.raises(ShortTenorReceiptError, match="access log"):
        validate_structural_training_receipt(receipt)


def test_training_rejects_run_order_or_unbound_access_log() -> None:
    receipt = _training_receipt()
    receipt["run_details"]["started_at_utc"] = "2025-12-31T00:00:00Z"
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="after plan freeze"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["run_details"]["finished_at_utc"] = "2028-01-02T00:00:00Z"
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="not after outer origin"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["run_details"]["byproduct_sha256s"]["access_log"] = _h("0")
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="access-log binding"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["run_details"]["byproduct_sha256s"]["inner_loss_commitment"] = _h("0")
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="inner-loss byproduct binding"):
        validate_structural_training_receipt(receipt)


def test_training_rejects_self_declared_authentication_or_authority_overlap() -> None:
    receipt = _training_receipt()
    receipt["authority"]["cryptographic_authentication_performed"] = True
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="cryptographic authentication"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["authority"]["trusted_time_key_id"] = receipt["authority"][
        "execution_key_id"
    ]
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="pairwise distinct"):
        validate_structural_training_receipt(receipt)

    receipt = _training_receipt()
    receipt["claims"]["model_input_authorized"] = True
    _resign(receipt)
    with pytest.raises(ShortTenorReceiptError, match="training claims"):
        validate_structural_training_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_receipt_id", _h("0"), "training id"),
        ("training_receipt_sha256", _h("0"), "training receipt hash"),
        ("candidate_grid_sha256", _h("0"), "candidate grid hash"),
        ("fold_inventory_sha256", _h("0"), "fold inventory hash"),
        ("outer_origin_entry_sha256", _h("1"), "outer origin entry hash"),
        ("inner_loss_commitment_sha256", _h("1"), "inner loss commitment hash"),
        ("selected_candidate_config_sha256", _h("f"), "outside training allowlist"),
        ("selection_rule", "LOWEST_LOSS", "selection rule"),
    ],
)
def test_selection_rejects_broken_training_grid_fold_candidate_or_rule_binding(
    field: str, value: object, message: str
) -> None:
    training = _training_receipt()
    selection = _selection_receipt(training)
    selection[field] = value
    _resign(selection)

    with pytest.raises(ShortTenorReceiptError, match=message):
        validate_structural_selection_receipt(selection, training_receipt=training)


def test_selection_rejects_subject_run_and_access_escalation() -> None:
    training = _training_receipt()
    selection = _selection_receipt(training)
    selection["subject"]["candidate_config_sha256"] = _h("e")
    _resign(selection)
    with pytest.raises(ShortTenorReceiptError, match="selection subject config"):
        validate_structural_selection_receipt(selection, training_receipt=training)

    selection = _selection_receipt(training)
    selection["run_details"]["started_at_utc"] = "2027-08-31T23:59:59Z"
    _resign(selection)
    with pytest.raises(ShortTenorReceiptError, match="follow training"):
        validate_structural_selection_receipt(selection, training_receipt=training)

    selection = _selection_receipt(training)
    selection["access_log"]["ompex_opened"] = True
    selection["run_details"]["byproduct_sha256s"]["access_log"] = _json_hash(
        selection["access_log"]
    )
    _resign(selection)
    with pytest.raises(ShortTenorReceiptError, match="selection access log"):
        validate_structural_selection_receipt(selection, training_receipt=training)


def test_receipt_boundary_remains_absent_from_lt_orchestration() -> None:
    source = inspect.getsource(production_phases)
    assert "eex_short_tenor_receipts" not in source
    assert "validate_structural_training_receipt" not in source
    assert "validate_structural_selection_receipt" not in source
