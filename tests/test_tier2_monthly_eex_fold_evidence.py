from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.calibration import tier2_monthly_eex_base_output_evidence as base_output
from pfc_shaping.calibration import tier2_monthly_eex_evaluation as tier2
from pfc_shaping.calibration import tier2_monthly_eex_fold_evidence as fold_evidence
from pfc_shaping.calibration import tier2_monthly_eex_selection_base_inputs as selection_inputs
from pfc_shaping.calibration import tier2_monthly_eex_selection_input_evidence as input_evidence
from pfc_shaping.calibration.tier2_monthly_eex_base_replay import (
    BASE_REPLAY_CONFIG_SCHEMA,
)
from pfc_shaping.calibration.tier2_monthly_eex_evaluation import (
    ProductConstraint,
    product_constraint_vector,
)
from pfc_shaping.data import ingest_forwards
from pfc_shaping.data.acquisition_signing import sign_acquisition_contract
from pfc_shaping.data.eex_historical_vintage import (
    EEX_HISTORICAL_CATALOG_SCHEMA,
    EEX_HISTORICAL_QUOTE_SCHEMA,
    historical_vintage_revision_id,
    historical_vintage_row_hash,
    historical_vintage_snapshot_id,
)
from pfc_shaping.pipeline.model_governance_contract import (
    MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
    sign_model_governance_artifact_receipt,
)


def test_signed_selection_fold_proves_data_lineage_not_model_output(
    governed_fold: dict[str, object],
) -> None:
    verified = _verify(governed_fold)

    assert verified.fold_id == governed_fold["fold_id"]
    assert verified.catalog_id == governed_fold["catalog_id"]
    assert verified.diagnostic_result_sha256 == hashlib.sha256(
        Path(governed_fold["result_path"]).read_bytes()
    ).hexdigest()
    assert not hasattr(verified, "metrics")
    diagnostic = json.loads(
        Path(governed_fold["result_path"]).read_text(encoding="utf-8")
    )
    assert diagnostic["data_lineage_verified"] is True
    assert diagnostic["model_output_verified"] is False
    assert diagnostic["campaign_eligible"] is False


def test_selection_base_input_derivation_is_config_neutral_and_premetric(
    governed_fold: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    d137_context = _verify_context(governed_fold)
    _, _, d137_retained_hashes, d137_history_hashes = base_output._base_inputs(
        d137_context
    )
    plan = tier2.verify_governed_plan_artifact(
        governed_fold["plan_path"],
        governance_receipt_path=governed_fold["plan_governance_path"],
        timestamp_receipt_path=governed_fold["plan_timestamp_path"],
    )
    frame, _, _ = fold_evidence._verify_eex_paths(
        Path(governed_fold["catalog_path"]),
        Path(governed_fold["history_path"]),
    )
    fold_spec = fold_evidence._find_selection_fold(plan.payload, str(governed_fold["fold_id"]))
    monkeypatch.setattr(
        fold_evidence,
        "_validate_result_payload",
        lambda *_, **__: pytest.fail("scored fold_result must not be read by D138"),
    )

    record = selection_inputs._derive_selection_base_input(
        frame,
        plan=plan,
        fold_spec=fold_spec,
    )
    payload = record.canonical_payload()

    assert record.target_load_type == "BASE"
    assert record.base_history_row_hashes
    assert record.base_retained_row_hashes
    assert record.base_history_row_hashes == d137_history_hashes
    assert record.base_retained_row_hashes == d137_retained_hashes
    assert record.delivery_months == d137_context.delivery_months
    assert record.base_input_set_sha256 == base_output.base_replay_module._base_input_set_sha256(
        d137_history_hashes,
        d137_retained_hashes,
    )
    forbidden = {
        "candidate_config_sha256",
        "target_price",
        "target_row_hash",
        "prediction",
        "metric",
        "rank",
        "winner",
    }
    serialized = json.dumps(payload, sort_keys=True)
    assert all(name not in serialized for name in forbidden)


def test_selection_base_input_preflight_preserves_exact_plan_order(
    governed_fold: dict[str, object],
) -> None:
    verified = tier2.verify_governed_plan_artifact(
        governed_fold["plan_path"],
        governance_receipt_path=governed_fold["plan_governance_path"],
        timestamp_receipt_path=governed_fold["plan_timestamp_path"],
    )
    selected = fold_evidence._find_selection_fold(
        verified.payload, str(governed_fold["fold_id"])
    )

    folds = selection_inputs._preflight_selection_folds(
        {"selection_folds": [selected]}
    )

    assert tuple(fold["fold_id"] for fold in folds) == (governed_fold["fold_id"],)


def test_selection_base_input_set_rejects_duck_typed_plan() -> None:
    plan = SimpleNamespace(
        plan_id="1" * 64,
        artifact_sha256="2" * 64,
        trusted_time_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
        payload={"selection_folds": []},
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="path-verified plan",
    ):
        selection_inputs._derive_selection_base_input_set(pd.DataFrame(), plan=plan)


def test_selection_base_input_rejects_fold_not_exactly_in_verified_plan(
    governed_fold: dict[str, object],
) -> None:
    plan = tier2.verify_governed_plan_artifact(
        governed_fold["plan_path"],
        governance_receipt_path=governed_fold["plan_governance_path"],
        timestamp_receipt_path=governed_fold["plan_timestamp_path"],
    )
    frame, _, _ = fold_evidence._verify_eex_paths(
        Path(governed_fold["catalog_path"]),
        Path(governed_fold["history_path"]),
    )
    registered = fold_evidence._find_selection_fold(
        plan.payload, str(governed_fold["fold_id"])
    )
    altered = dict(registered)
    altered["target_product"] = "2027-03"

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="exact preregistered",
    ):
        selection_inputs._derive_selection_base_input(
            frame,
            plan=plan,
            fold_spec=altered,
        )


def test_selection_base_input_set_rejects_non_base_before_partial_derivation(
) -> None:
    folds = [
        {
            "fold_id": "base-fold",
            "target_load_type": "BASE",
        },
        {
            "fold_id": "peak-fold",
            "target_load_type": "PEAK",
        },
    ]
    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match=selection_inputs.UNSUPPORTED_NON_BASE_SELECTION,
    ):
        selection_inputs._preflight_selection_folds({"selection_folds": folds})


def test_selection_base_input_set_enforces_fold_resource_bound() -> None:
    payload = {
            "selection_folds": [
                {"fold_id": f"fold-{index}", "target_load_type": "BASE"}
                for index in range(selection_inputs.MAX_SELECTION_FOLDS + 1)
            ]
        }

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="resource bounds",
    ):
        selection_inputs._preflight_selection_folds(payload)


def test_selection_base_input_set_rejects_hash_work_before_derivation(
    governed_fold: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tier2.verify_governed_plan_artifact(
        governed_fold["plan_path"],
        governance_receipt_path=governed_fold["plan_governance_path"],
        timestamp_receipt_path=governed_fold["plan_timestamp_path"],
    )
    frame, _, _ = fold_evidence._verify_eex_paths(
        Path(governed_fold["catalog_path"]),
        Path(governed_fold["history_path"]),
    )
    monkeypatch.setattr(selection_inputs, "MAX_TOTAL_EMITTED_ROW_HASHES", 1)
    monkeypatch.setattr(
        selection_inputs,
        "_derive_solver_input_frames",
        lambda *_, **__: pytest.fail("resource rejection must precede derivation"),
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="emitted row hashes",
    ):
        selection_inputs._derive_selection_base_input_set(frame, plan=plan)


def test_selection_base_input_set_derives_complete_governed_fold_inventory(
    governed_selection_input_set: dict[str, object],
) -> None:
    plan = tier2.verify_governed_plan_artifact(
        governed_selection_input_set["plan_path"],
        governance_receipt_path=governed_selection_input_set["plan_governance_path"],
        timestamp_receipt_path=governed_selection_input_set["plan_timestamp_path"],
    )
    frame, _, _ = fold_evidence._verify_eex_paths(
        Path(governed_selection_input_set["catalog_path"]),
        Path(governed_selection_input_set["history_path"]),
    )

    derived = selection_inputs._derive_selection_base_input_set(frame, plan=plan)
    expected_ids = tuple(
        str(fold["fold_id"]) for fold in plan.payload["selection_folds"]
    )

    assert derived.fold_count == len(expected_ids)
    assert derived.ordered_fold_ids == expected_ids
    assert tuple(record.fold_id for record in derived.records) == expected_ids
    assert all(record.target_load_type == "BASE" for record in derived.records)
    serialized = json.dumps(derived.canonical_payload(), sort_keys=True)
    for forbidden in (
        "candidate_config_sha256",
        "target_price",
        "target_row_hash",
        "prediction",
        "metric",
        "winner",
    ):
        assert forbidden not in serialized


def test_signed_selection_base_inputs_replay_complete_config_neutral_inventory(
    governed_selection_input_set: dict[str, object],
) -> None:
    verified = input_evidence.verify_signed_selection_base_inputs(
        plan_path=governed_selection_input_set["plan_path"],
        plan_governance_receipt_path=governed_selection_input_set[
            "plan_governance_path"
        ],
        plan_timestamp_receipt_path=governed_selection_input_set[
            "plan_timestamp_path"
        ],
        eex_catalog_path=governed_selection_input_set["catalog_path"],
        eex_history_path=governed_selection_input_set["history_path"],
        selection_input_manifest_path=governed_selection_input_set[
            "selection_input_manifest_path"
        ],
        execution_attestation_path=governed_selection_input_set[
            "selection_input_execution_path"
        ],
        linked_timestamp_receipt_path=governed_selection_input_set[
            "selection_input_time_path"
        ],
        journal_head_path=governed_selection_input_set["journal_head_path"],
        source_closure_path=governed_selection_input_set["source_closure_path"],
        runtime_lock_path=governed_selection_input_set["runtime_lock_path"],
    )

    plan = json.loads(
        Path(governed_selection_input_set["plan_path"]).read_text(encoding="utf-8")
    )
    assert verified.fold_count == len(plan["selection_folds"])
    assert verified.configuration_neutral_verified is True
    assert verified.model_output_verified is False
    assert verified.configuration_selection_verified is False
    assert verified.target_non_observation_verified is False
    assert verified.metrics_verified is False
    assert verified.campaign_eligible is False
    assert verified.production_approved is False


def test_signed_selection_base_inputs_rejects_mutated_payload(
    governed_selection_input_set: dict[str, object],
) -> None:
    payload_path = (
        Path(governed_selection_input_set["selection_input_manifest_path"]).parent
        / "selection_base_inputs.v1.json"
    )
    payload_path.write_bytes(payload_path.read_bytes() + b"\n")

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="replayed selection input bytes"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_resigned_missing_fold(
    governed_selection_input_set: dict[str, object],
) -> None:
    payload_path = (
        Path(governed_selection_input_set["selection_input_manifest_path"]).parent
        / "selection_base_inputs.v1.json"
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["records"].pop()
    payload["ordered_fold_ids"].pop()
    payload["fold_count"] = len(payload["records"])
    payload.pop("input_set_id")
    payload["input_set_id"] = tier2._sha256_json(payload)
    _rebind_selection_input_payload(governed_selection_input_set, payload)

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="replayed selection input bytes"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_extra_package_file(
    governed_selection_input_set: dict[str, object],
) -> None:
    package = Path(governed_selection_input_set["selection_input_manifest_path"]).parent
    (package / "stale.tmp").write_text("stale", encoding="utf-8")

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="exact inventory"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_legacy_plan_v1(
    governed_selection_input_set: dict[str, object],
) -> None:
    plan_path = Path(governed_selection_input_set["plan_path"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["schema_version"] = tier2.PLAN_SCHEMA
    plan.pop("selection_input_resource_profile")
    _repin(plan, "plan_id")
    _canonical_write(plan_path, plan)
    _canonical_write(
        Path(governed_selection_input_set["plan_governance_path"]),
        sign_model_governance_artifact_receipt(
            plan_path,
            artifact_role="tier2_monthly_eex_evaluation_plan",
            authority_id="MODEL-RISK",
            issued_at_utc="2026-01-01T08:00:00Z",
            data_cutoff_utc="2025-12-31T23:59:59Z",
            private_key_path=Path(
                governed_selection_input_set["governance_private"]
            ),
        ),
    )
    _canonical_write(
        Path(governed_selection_input_set["plan_timestamp_path"]),
        _artifact_time_receipt(
            plan_path,
            Path(governed_selection_input_set["tier2_time_private"]),
            role="tier2_monthly_eex_evaluation_plan",
            artifact_schema=tier2.PLAN_SCHEMA,
            received_at="2026-01-01T08:00:01Z",
        ),
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="D138 plan schema"):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_cannot_self_declare_production(
    governed_selection_input_set: dict[str, object],
) -> None:
    manifest_path = Path(
        governed_selection_input_set["selection_input_manifest_path"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["production_approved"] = True
    _repin(manifest, "manifest_id")
    _canonical_write(manifest_path, manifest)
    _resign_selection_input_chain(governed_selection_input_set)

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="selection input production_approved",
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_unlinked_execution_attestation(
    governed_selection_input_set: dict[str, object],
) -> None:
    time_path = Path(governed_selection_input_set["selection_input_time_path"])
    payload = json.loads(time_path.read_text(encoding="utf-8"))
    payload.pop("trusted_time_attestation")
    payload["execution_attestation_sha256"] = "0" * 64
    payload.pop("receipt_id")
    payload["receipt_id"] = tier2._sha256_json(payload)
    private = serialization.load_pem_private_key(
        Path(governed_selection_input_set["tier2_time_private"]).read_bytes(),
        password=None,
    )
    _canonical_write(
        time_path,
        _sign_payload(
            payload,
            private,
            signature_field="trusted_time_attestation",
            id_payload_sha=True,
        ),
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="linked execution attestation hash"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_plan_receipt_aba_substitution(
    governed_selection_input_set: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substituted_path = tmp_path / "substituted-plan-time.json"
    _canonical_write(
        substituted_path,
        _artifact_time_receipt(
            Path(governed_selection_input_set["plan_path"]),
            Path(governed_selection_input_set["tier2_time_private"]),
            role="tier2_monthly_eex_evaluation_plan",
            artifact_schema=tier2.SELECTION_INPUT_PLAN_SCHEMA,
            received_at="2026-01-01T08:00:01.500000Z",
        ),
    )
    original_load = tier2._load_json_artifact

    def substitute(path: object) -> tuple[Path, dict[str, object], bytes]:
        resolved = Path(path).resolve()
        if resolved == Path(governed_selection_input_set["plan_timestamp_path"]).resolve():
            return original_load(substituted_path)
        return original_load(path)

    monkeypatch.setattr(tier2, "_load_json_artifact", substitute)

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="captured plan timestamp receipt hash",
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_journal_gap(
    governed_selection_input_set: dict[str, object],
) -> None:
    time_path = Path(governed_selection_input_set["selection_input_time_path"])
    payload = json.loads(time_path.read_text(encoding="utf-8"))
    payload.pop("trusted_time_attestation")
    payload["journal_sequence"] = int(payload["journal_sequence"]) + 1
    payload.pop("receipt_id")
    payload["receipt_id"] = tier2._sha256_json(payload)
    private = serialization.load_pem_private_key(
        Path(governed_selection_input_set["tier2_time_private"]).read_bytes(),
        password=None,
    )
    _canonical_write(
        time_path,
        _sign_payload(
            payload,
            private,
            signature_field="trusted_time_attestation",
            id_payload_sha=True,
        ),
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="linked timestamp sequence"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_foreign_journal_head(
    governed_selection_input_set: dict[str, object],
) -> None:
    head_path = Path(governed_selection_input_set["journal_head_path"])
    payload = json.loads(head_path.read_text(encoding="utf-8"))
    payload.pop("witness_attestation")
    payload["entries"][1]["receipt_id"] = "0" * 64
    entry = dict(payload["entries"][1])
    entry.pop("head_id")
    payload["entries"][1]["head_id"] = tier2._sha256_json(entry)
    payload["current_head_id"] = payload["entries"][1]["head_id"]
    payload.pop("journal_head_id")
    payload["journal_head_id"] = tier2._sha256_json(payload)
    private = serialization.load_pem_private_key(
        Path(governed_selection_input_set["journal_witness_private"]).read_bytes(),
        password=None,
    )
    _canonical_write(
        head_path,
        _sign_payload(
            payload,
            private,
            signature_field="witness_attestation",
            id_payload_sha=True,
        ),
    )
    os.environ[input_evidence.TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV] = str(
        payload["current_head_id"]
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="does not include the linked D138 receipt",
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_caller_selected_old_head_path(
    governed_selection_input_set: dict[str, object],
    tmp_path: Path,
) -> None:
    original = Path(governed_selection_input_set["journal_head_path"])
    old_head = tmp_path / "old-signed-head.json"
    old_head.write_bytes(original.read_bytes())
    governed_selection_input_set["journal_head_path"] = old_head

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="trusted timestamp journal head path",
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_remains_auditable_after_journal_progression(
    governed_selection_input_set: dict[str, object],
) -> None:
    _append_witnessed_journal_entry(governed_selection_input_set)

    verified = _verify_selection_inputs(governed_selection_input_set)

    assert verified.data_lineage_verified is True


def test_signed_selection_base_inputs_rejects_same_path_journal_rollback(
    governed_selection_input_set: dict[str, object],
) -> None:
    head_path = Path(governed_selection_input_set["journal_head_path"])
    old_bytes = head_path.read_bytes()
    _append_witnessed_journal_entry(governed_selection_input_set)
    head_path.write_bytes(old_bytes)

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="external current journal head ID",
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_noncanonical_envelope_bytes(
    governed_selection_input_set: dict[str, object],
) -> None:
    execution_path = Path(
        governed_selection_input_set["selection_input_execution_path"]
    )
    execution_path.write_bytes(execution_path.read_bytes() + b"\n")

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="canonical UTF-8"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_changed_source_journal(
    governed_selection_input_set: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_JOURNAL_ID", "different-source-journal")

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="EEX source verification failed"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_preflights_payload_byte_bound(
    governed_selection_input_set: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_path = (
        Path(governed_selection_input_set["selection_input_manifest_path"]).parent
        / "selection_base_inputs.v1.json"
    )
    monkeypatch.setitem(
        tier2.EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE,
        "max_selection_input_payload_bytes",
        payload_path.stat().st_size - 1,
    )
    monkeypatch.setattr(
        input_evidence,
        "_package_snapshot",
        lambda *_args, **_kwargs: pytest.fail("preflight must precede package hashing"),
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="payload exceeds the preregistered byte-size bound",
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_signed_selection_base_inputs_rejects_loaded_derivation_substitution(
    governed_selection_input_set: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_inputs,
        "_derive_selection_base_input_set",
        lambda *_args, **_kwargs: pytest.fail("substituted derivation executed"),
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError, match="source closure files"
    ):
        _verify_selection_inputs(governed_selection_input_set)


def test_fold_package_rejects_mutated_rows_bytes(
    governed_fold: dict[str, object],
) -> None:
    rows_path = Path(governed_fold["manifest_path"]).parent / "fold_rows.v1.json"
    rows_path.write_bytes(rows_path.read_bytes() + b"\n")

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="canonical UTF-8"):
        _verify(governed_fold)


def test_fold_package_rejects_extra_file(
    governed_fold: dict[str, object],
) -> None:
    package = Path(governed_fold["manifest_path"]).parent
    (package / "stale.json").write_text("{}", encoding="utf-8")

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="exact inventory"):
        _verify(governed_fold)


def test_fold_rows_cannot_add_masked_target_to_permitted_inputs(
    governed_fold: dict[str, object],
) -> None:
    package = Path(governed_fold["manifest_path"]).parent
    rows_path = package / "fold_rows.v1.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    target_hash = rows["roles"]["masked_target"]["row_hashes"][0]
    rows["permitted_candidate_input_row_hashes"].append(target_hash)
    _canonical_write(rows_path, rows)
    _rebind_package_and_receipts(governed_fold)

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="permitted candidate input row",
    ):
        _verify(governed_fold)


def test_fold_rows_cannot_omit_future_diagnostic_inventory(
    governed_fold: dict[str, object],
) -> None:
    rows_path = Path(governed_fold["manifest_path"]).parent / "fold_rows.v1.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    columns = [
        "available_at",
        "snapshot_id",
        "market",
        "load_type",
        "product",
        "row_hash",
        "quote_id",
        "revision_id",
        "source_document_id",
        "source_document_sha256",
        "acquisition_id",
    ]
    rows["roles"]["diagnostic_later_quotes"] = fold_evidence._inventory(
        "diagnostic_later_quotes", pd.DataFrame(columns=columns)
    )
    _canonical_write(rows_path, rows)
    _rebind_package_and_receipts(governed_fold)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="role inventories"):
        _verify(governed_fold)


def test_fold_result_metrics_are_recomputed_not_trusted(
    governed_fold: dict[str, object],
) -> None:
    result_path = Path(governed_fold["manifest_path"]).parent / "fold_result.v1.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["reported_metrics"]["candidate_prediction"] = 999.0
    _canonical_write(result_path, result)
    _rebind_package_and_receipts(governed_fold)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="candidate_prediction"):
        _verify(governed_fold)


@pytest.mark.parametrize("authority", ["acquisition", "source-time"])
def test_execution_authority_cannot_reuse_source_authority(
    governed_fold: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    monkeypatch.setenv(
        tier2.EXECUTION_PUBLIC_KEY_ENV,
        str(governed_fold[f"{authority}_public"]),
    )
    _canonical_write(
        Path(governed_fold["execution_path"]),
        _execution_attestation(
            Path(governed_fold["manifest_path"]),
            Path(governed_fold[f"{authority}_private"]),
            executed_at="2026-01-02T08:00:00Z",
        ),
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="authorit"):
        _verify(governed_fold)


def test_eex_trust_anchor_must_be_absolute(
    governed_fold: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        Path(governed_fold["acquisition_public"]).name,
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="must be absolute"):
        _verify(governed_fold)


def test_fold_package_change_during_verification_is_rejected(
    governed_fold: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fold_evidence._validate_result_payload

    def mutate_package(*args: object, **kwargs: object) -> dict[str, object]:
        result = original(*args, **kwargs)
        package = Path(governed_fold["manifest_path"]).parent
        (package / "late.json").write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(fold_evidence, "_validate_result_payload", mutate_package)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="exact inventory"):
        _verify(governed_fold)


def test_fold_package_change_during_private_record_materialization_is_rejected(
    governed_fold: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fold_evidence._replay_rows
    mutated = False

    def mutate_package(*args: object, **kwargs: object):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            package = Path(governed_fold["manifest_path"]).parent
            (package / "late-during-records.json").write_text("{}", encoding="utf-8")
            mutated = True
        return result

    monkeypatch.setattr(fold_evidence, "_replay_rows", mutate_package)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="exact inventory"):
        _verify_context(governed_fold)


def test_manifest_change_after_read_before_snapshot_is_rejected(
    governed_fold: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = Path(governed_fold["manifest_path"])
    original = fold_evidence._load_json_artifact
    mutated = False

    def mutate_after_read(path: str | Path) -> tuple[Path, dict[str, object], bytes]:
        nonlocal mutated
        loaded = original(path)
        if Path(path).resolve() == manifest_path.resolve() and not mutated:
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            persisted["model_output_verified"] = True
            persisted["campaign_eligible"] = True
            _repin(persisted, "evidence_id")
            _canonical_write(manifest_path, persisted)
            mutated = True
        return loaded

    monkeypatch.setattr(fold_evidence, "_load_json_artifact", mutate_after_read)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="changed during"):
        _verify(governed_fold)


def test_eex_internal_file_change_during_verification_is_rejected(
    governed_fold: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fold_evidence.verify_eex_historical_vintage_catalog

    def mutate_source(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        source_path = Path(governed_fold["source_document_path"])
        source_path.write_bytes(source_path.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(
        fold_evidence,
        "verify_eex_historical_vintage_catalog",
        mutate_source,
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="changed during"):
        _verify(governed_fold)


def test_candidate_grid_bytes_are_bound_by_signed_plan(
    governed_fold: dict[str, object],
) -> None:
    grid_path = Path(governed_fold["grid_path"])
    grid_path.write_bytes(grid_path.read_bytes() + b"\n")

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="canonical UTF-8"):
        _verify(governed_fold)


def test_fold_manifest_rejects_duplicate_json_keys(
    governed_fold: dict[str, object],
) -> None:
    manifest_path = Path(governed_fold["manifest_path"])
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        '{"schema_version":"duplicate",' + original[1:], encoding="utf-8"
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="duplicate JSON key"):
        _verify(governed_fold)


def test_fold_manifest_rejects_holdout_without_freeze(
    governed_fold: dict[str, object],
) -> None:
    plan = json.loads(Path(governed_fold["plan_path"]).read_text(encoding="utf-8"))
    holdout = plan["holdout_folds"][0]
    manifest_path = Path(governed_fold["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in (
        "fold_id",
        "fold_spec_hash",
        "campaign_type",
        "origin_id",
        "fold_as_of_utc",
    ):
        manifest[field] = holdout[field]
    _repin(manifest, "evidence_id")
    _canonical_write(manifest_path, manifest)
    _resign_fold_receipts(governed_fold)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="HOLDOUT_REQUIRES"):
        _verify(governed_fold)


def test_verified_fold_proof_cannot_be_constructed_directly() -> None:
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="path-based"):
        fold_evidence.VerifiedSelectionFoldDataLineage(
            evidence_id="1" * 64,
            artifact_sha256="2" * 64,
            fold_id="fold",
            fold_spec_hash="3" * 64,
            candidate_config_sha256="4" * 64,
            catalog_id="5" * 64,
            frame_semantic_sha256="6" * 64,
            execution_key_id="7" * 64,
            timestamp_key_id="8" * 64,
            executed_at_utc=pd.Timestamp("2026-01-02T00:00:00Z"),
            trusted_time_utc=pd.Timestamp("2026-01-02T00:00:01Z"),
            diagnostic_result_sha256="9" * 64,
        )


def test_private_replay_context_uses_same_verified_snapshot_and_is_immutable(
    governed_fold: dict[str, object],
) -> None:
    context = _verify_context(governed_fold)
    public = _verify(governed_fold)
    rows = json.loads(
        (Path(governed_fold["manifest_path"]).parent / "fold_rows.v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert context.proof == public
    assert [row.row_hash for row in context.historical_rows] == rows["roles"][
        "fold_historical_context"
    ]["row_hashes"]
    assert [row.row_hash for row in context.retained_rows] == rows["roles"][
        "fold_evaluation_snapshot_retained"
    ]["row_hashes"]
    with pytest.raises(TypeError):
        context.candidate_grid["market"] = "DE"  # type: ignore[index]


def test_private_replay_context_cannot_be_constructed_directly(
    governed_fold: dict[str, object],
) -> None:
    proof = _verify(governed_fold)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="same-snapshot"):
        fold_evidence._VerifiedSelectionFoldReplayContext(
            proof=proof,
            plan_id="a" * 64,
            plan_artifact_sha256="b" * 64,
            candidate_grid_sha256="c" * 64,
            evaluation_source_closure_sha256="d" * 64,
            runtime_lock_sha256="e" * 64,
            plan_trusted_time_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
            plan_governance_receipt_sha256="f" * 64,
            plan_timestamp_receipt_sha256="0" * 64,
            fold_execution_receipt_sha256="1" * 64,
            fold_timestamp_receipt_sha256="2" * 64,
            target_load_type="BASE",
            plan_governance_key_id="3" * 64,
            plan_timestamp_key_id="4" * 64,
            fold_execution_key_id="5" * 64,
            fold_timestamp_key_id="6" * 64,
            acquisition_key_id="7" * 64,
            source_timestamp_key_ids=("8" * 64,),
            forbidden_non_time_authority_key_ids=(),
            trusted_time_authority_key_ids=(),
            historical_rows=(),
            retained_rows=(),
            delivery_months=(),
            candidate_grid={},
            catalog_sha256="1" * 64,
            history_sha256="2" * 64,
        )


def test_signed_base_output_is_replayed_byte_for_byte_without_metrics(
    governed_base_output: dict[str, object],
) -> None:
    verified = _verify_base_output(governed_base_output)

    assert verified.model_output_verified is True
    assert verified.data_lineage_verified is True
    assert verified.metrics_verified is False
    assert verified.configuration_selection_verified is False
    assert verified.process_isolation_verified is False
    assert verified.campaign_eligible is False
    assert verified.production_approved is False
    assert verified.peak_offpeak_verified is False
    assert verified.model_execution_receipt_sha256 == hashlib.sha256(
        Path(governed_base_output["model_execution_path"]).read_bytes()
    ).hexdigest()
    assert verified.timestamp_receipt_sha256 == hashlib.sha256(
        Path(governed_base_output["output_timestamp_path"]).read_bytes()
    ).hexdigest()
    assert not hasattr(verified, "candidate_monthly_base")


def test_base_output_manifest_cannot_self_declare_model_verified(
    governed_base_output: dict[str, object],
) -> None:
    path = Path(governed_base_output["base_output_manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model_output_verified"] = True
    _repin(manifest, "manifest_id")
    _canonical_write(path, manifest)
    _resign_base_output_receipts(governed_base_output)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="model_output_verified"):
        _verify_base_output(governed_base_output)


def test_base_output_core_receives_only_permitted_ch_base_rows(
    governed_base_output: dict[str, object],
) -> None:
    context = _verify_context(governed_base_output)
    retained, history, _, _ = base_output._base_inputs(context)
    fold_rows = json.loads(Path(governed_base_output["rows_path"]).read_text(encoding="utf-8"))
    forbidden = {
        *fold_rows["roles"]["masked_target"]["row_hashes"],
        *fold_rows["roles"]["removed_revealing_quotes"]["row_hashes"],
        *fold_rows["roles"]["diagnostic_later_quotes"]["row_hashes"],
    }

    assert set(retained["market"]) == {"CH"}
    assert set(retained["load_type"]) == {"BASE"}
    assert set(history["market"]) == {"CH"}
    assert set(history["load_type"]) == {"BASE"}
    assert forbidden.isdisjoint(retained["row_hash"])
    assert forbidden.isdisjoint(history["row_hash"])

    assert _verify_base_output(governed_base_output).model_output_verified is True


def test_base_output_loaded_solver_substitution_is_rejected(
    governed_base_output: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = base_output.base_replay_module._replay_monthly_base_outputs

    def substituted_solver(*args: object, **kwargs: object):
        return original(*args, **kwargs)

    monkeypatch.setattr(
        base_output.base_replay_module,
        "_replay_monthly_base_outputs",
        substituted_solver,
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="source closure files"):
        _verify_base_output(governed_base_output)


def test_base_output_loaded_internal_solver_alias_substitution_is_rejected(
    governed_base_output: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = (
        base_output.base_replay_module.solve_monthly_forward_curve_from_constraints
    )

    def substituted_internal_solver(*args: object, **kwargs: object):
        return original(*args, **kwargs)

    monkeypatch.setattr(
        base_output.base_replay_module,
        "solve_monthly_forward_curve_from_constraints",
        substituted_internal_solver,
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="source closure files"):
        _verify_base_output(governed_base_output)


def test_base_output_loaded_imported_class_spoof_is_rejected(
    governed_base_output: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = base_output.base_replay_module.MonthlyCurveConfig
    fake = type("MonthlyCurveConfig", (), {})
    fake.__module__ = original.__module__
    fake.__qualname__ = original.__qualname__
    monkeypatch.setattr(base_output.base_replay_module, "MonthlyCurveConfig", fake)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="source closure files"):
        _verify_base_output(governed_base_output)


def test_base_output_loaded_governed_regex_substitution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = base_output._loaded_module_code_sha256(
        base_output.monthly_solver_module
    )
    monkeypatch.setattr(base_output.monthly_solver_module, "_MONTH_RE", re.compile("^MUTATED$"))

    assert (
        base_output._loaded_module_code_sha256(base_output.monthly_solver_module)
        != before
    )


def test_base_output_loaded_module_spoof_is_rejected(
    governed_base_output: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_math = SimpleNamespace(
        __name__="math",
        isfinite=lambda _: True,
    )
    monkeypatch.setattr(base_output.base_replay_module, "math", fake_math)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="source closure files"):
        _verify_base_output(governed_base_output)


def test_base_output_package_inventory_mutation_is_rejected(
    governed_base_output: dict[str, object],
) -> None:
    package = Path(governed_base_output["base_output_manifest_path"]).parent
    (package / "late.json").write_text("{}", encoding="utf-8")

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="exact inventory"):
        _verify_base_output(governed_base_output)


def test_base_output_rejects_fold_execution_key_as_model_execution_key(
    governed_base_output: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        base_output.MODEL_EXECUTION_PUBLIC_KEY_ENV,
        str(governed_base_output["execution_public"]),
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="pairwise distinct"):
        _verify_base_output(governed_base_output)


def test_verified_base_output_token_cannot_be_constructed_directly() -> None:
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="path-only"):
        base_output.VerifiedSelectionFoldBaseModelOutput(
            manifest_id="1" * 64,
            manifest_sha256="2" * 64,
            lineage_evidence_id="3" * 64,
            fold_id="fold",
            candidate_config_sha256="4" * 64,
            core_output_sha256="5" * 64,
            source_closure_sha256="6" * 64,
            runtime_lock_sha256="7" * 64,
            model_execution_key_id="8" * 64,
            timestamp_key_id="9" * 64,
            model_executed_at_utc=pd.Timestamp("2026-01-02T00:00:00Z"),
            trusted_time_utc=pd.Timestamp("2026-01-02T00:00:01Z"),
        )


def test_verified_base_output_claims_cannot_be_overridden() -> None:
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="path-only"):
        base_output.VerifiedSelectionFoldBaseModelOutput(
            manifest_id="1" * 64,
            manifest_sha256="2" * 64,
            lineage_evidence_id="3" * 64,
            fold_id="fold",
            candidate_config_sha256="4" * 64,
            core_output_sha256="5" * 64,
            source_closure_sha256="6" * 64,
            runtime_lock_sha256="7" * 64,
            model_execution_key_id="8" * 64,
            timestamp_key_id="9" * 64,
            model_executed_at_utc=pd.Timestamp("2026-01-02T00:00:00Z"),
            trusted_time_utc=pd.Timestamp("2026-01-02T00:00:01Z"),
            campaign_eligible=True,
        )
    assert not hasattr(base_output, "_VERIFIED_OUTPUT_TOKEN")
    assert not hasattr(base_output, "_create_verified_output")


def test_base_output_rejects_temporary_signed_execution_receipt_substitution(
    governed_base_output: dict[str, object],
) -> None:
    original_path = Path(governed_base_output["model_execution_path"])
    alternate_path = original_path.with_name("alternate-model-execution.json")
    _canonical_write(
        alternate_path,
        _execution_attestation(
            Path(governed_base_output["base_output_manifest_path"]),
            Path(governed_base_output["model_execution_private"]),
            executed_at="2026-01-02T08:00:02.500000Z",
            role=base_output.BASE_OUTPUT_MANIFEST_ROLE,
            artifact_schema=base_output.BASE_OUTPUT_MANIFEST_SCHEMA,
        ),
    )
    _, _, alternate_bytes = base_output._load_json_artifact(alternate_path)
    initial_snapshot = {
        str(original_path): base_output._file_snapshot(original_path)
    }

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="initial snapshot hash",
    ):
        base_output._bind_read_bytes_to_snapshot(
            original_path,
            alternate_bytes,
            initial_snapshot,
        )


def test_base_output_rejects_context_not_bound_to_captured_outer_bytes(
    governed_base_output: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _verify_context(governed_base_output)
    forged_fields = dict(vars(context))
    forged_fields["plan_artifact_sha256"] = "f" * 64
    forged = SimpleNamespace(**forged_fields)
    monkeypatch.setattr(
        base_output,
        "_verify_signed_selection_fold_evidence_context",
        lambda **_: forged,
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="consumed plan"):
        _verify_base_output(governed_base_output)


def test_base_output_rejects_role_permutation_with_unchanged_authority_sets(
    governed_base_output: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _verify_context(governed_base_output)
    permutations = (
        ("plan_governance_key_id", "acquisition_key_id", "plan governance"),
        ("plan_timestamp_key_id", "source_timestamp_key_ids", "plan timestamp"),
    )
    for left, right, expected_label in permutations:
        fields = dict(vars(context))
        if right == "source_timestamp_key_ids":
            source_key = fields[right][0]
            fields[right] = (fields[left],)
            fields[left] = source_key
        else:
            fields[left], fields[right] = fields[right], fields[left]
        forged = SimpleNamespace(**fields)
        monkeypatch.setattr(
            base_output,
            "_verify_signed_selection_fold_evidence_context",
            lambda **_: forged,
        )

        with pytest.raises(
            tier2.Tier2MonthlyEexEvaluationError,
            match=expected_label,
        ):
            _verify_base_output(governed_base_output)


def test_base_output_rejects_authority_not_equal_to_captured_trust_anchor(
    governed_base_output: dict[str, object],
) -> None:
    context = _verify_context(governed_base_output)
    trust_policy = base_output._capture_trust_policy()
    execution = SimpleNamespace(authority_key_id="f" * 64)
    trusted_time = SimpleNamespace(
        authority_key_id=trust_policy["key_ids"][tier2.TIMESTAMP_PUBLIC_KEY_ENV]
    )

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="captured model execution authority",
    ):
        base_output._validate_authorities_and_time(
            context,
            execution,
            trusted_time,
            trust_policy=trust_policy,
        )


@pytest.fixture
def governed_base_output(
    governed_fold: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    root = tmp_path / "base-output-fixture"
    root.mkdir()
    catalog_path, history_path, catalog_root = _build_signed_eex_catalog(
        root,
        acquisition_private=Path(governed_fold["acquisition_private"]),
        source_time_private=Path(governed_fold["source-time_private"]),
        rich_history=True,
    )
    source_closure_path = _write_source_closure(root)
    runtime_lock_path = _write_runtime_lock(root)
    plan = json.loads(Path(governed_fold["plan_path"]).read_text(encoding="utf-8"))
    plan["selection_eex_root"] = catalog_root
    plan["evaluation_source_closure_sha256"] = hashlib.sha256(
        source_closure_path.read_bytes()
    ).hexdigest()
    plan["runtime_lock_sha256"] = hashlib.sha256(
        runtime_lock_path.read_bytes()
    ).hexdigest()
    _repin(plan, "plan_id")
    plan_path = root / "plan.json"
    _canonical_write(plan_path, plan)
    plan_governance_path = root / "plan-governance.json"
    plan_timestamp_path = root / "plan-time.json"
    _canonical_write(
        plan_governance_path,
        sign_model_governance_artifact_receipt(
            plan_path,
            artifact_role="tier2_monthly_eex_evaluation_plan",
            authority_id="MODEL-RISK",
            issued_at_utc="2026-01-01T08:00:00Z",
            data_cutoff_utc="2025-12-31T23:59:59Z",
            private_key_path=Path(governed_fold["governance_private"]),
        ),
    )
    _canonical_write(
        plan_timestamp_path,
        _artifact_time_receipt(
            plan_path,
            Path(governed_fold["tier2_time_private"]),
            role="tier2_monthly_eex_evaluation_plan",
            artifact_schema=tier2.PLAN_SCHEMA,
            received_at="2026-01-01T08:00:01Z",
        ),
    )
    fold_spec = next(
        fold
        for fold in plan["selection_folds"]
        if fold["fold_as_of_utc"] == "2025-03-15T08:00:00+00:00"
        and fold["tenor_horizon_bucket"] == "MONTH_H2"
    )
    selected_hash = str(
        json.loads(Path(governed_fold["manifest_path"]).read_text(encoding="utf-8"))[
            "candidate_config_sha256"
        ]
    )
    manifest_path, rows_path, result_path = _build_fold_package(
        root,
        plan=plan,
        fold_spec=fold_spec,
        grid_path=Path(governed_fold["grid_path"]),
        candidate_config_hash=selected_hash,
        catalog_path=catalog_path,
        history_path=history_path,
    )
    fold_execution_path = root / "fold-execution.json"
    fold_timestamp_path = root / "fold-time.json"
    _canonical_write(
        fold_execution_path,
        _execution_attestation(
            manifest_path,
            Path(governed_fold["execution_private"]),
            executed_at="2026-01-02T08:00:00Z",
        ),
    )
    _canonical_write(
        fold_timestamp_path,
        _artifact_time_receipt(
            manifest_path,
            Path(governed_fold["tier2_time_private"]),
            role=fold_evidence.FOLD_MANIFEST_ROLE,
            artifact_schema=fold_evidence.FOLD_MANIFEST_SCHEMA,
            received_at="2026-01-02T08:00:01Z",
        ),
    )
    paths: dict[str, object] = {
        **governed_fold,
        "plan_path": plan_path,
        "plan_governance_path": plan_governance_path,
        "plan_timestamp_path": plan_timestamp_path,
        "manifest_path": manifest_path,
        "rows_path": rows_path,
        "result_path": result_path,
        "execution_path": fold_execution_path,
        "fold_timestamp_path": fold_timestamp_path,
        "catalog_path": catalog_path,
        "history_path": history_path,
        "source_closure_path": source_closure_path,
        "runtime_lock_path": runtime_lock_path,
    }
    context = _verify_context(paths)
    config = base_output._selected_candidate_config(context)
    retained, history, retained_hashes, history_hashes = base_output._base_inputs(context)
    output = base_output.base_replay_module._replay_monthly_base_outputs(
        delivery_months=context.delivery_months,
        retained_quotes=retained,
        fold_historical_context=history,
        candidate_config=config,
    )
    output_package = root / "base-model-output"
    output_package.mkdir()
    expected_path = output_package / "expected_base_output.v1.json"
    _canonical_write(expected_path, output.canonical_payload())
    expected_bytes = expected_path.read_bytes()
    output_manifest: dict[str, object] = {
        "schema_version": base_output.BASE_OUTPUT_MANIFEST_SCHEMA,
        "artifact_role": base_output.BASE_OUTPUT_MANIFEST_ROLE,
        "scope": base_output.BASE_OUTPUT_SCOPE,
        "full_tier2_pfc_status": tier2.FULL_TIER2_STATUS,
        "status": base_output.AWAITING_REPLAY_STATUS,
        "data_lineage_verified": True,
        "model_output_verified": False,
        "metrics_verified": False,
        "campaign_eligible": False,
        "production_approved": False,
        "peak_offpeak_verified": False,
        "plan_id": context.plan_id,
        "plan_artifact_sha256": context.plan_artifact_sha256,
        "fold_id": context.proof.fold_id,
        "fold_spec_hash": context.proof.fold_spec_hash,
        "lineage_evidence_id": context.proof.evidence_id,
        "fold_manifest_sha256": context.proof.artifact_sha256,
        "catalog_id": context.proof.catalog_id,
        "catalog_sha256": context.catalog_sha256,
        "history_sha256": context.history_sha256,
        "frame_semantic_sha256": context.proof.frame_semantic_sha256,
        "candidate_grid_sha256": context.candidate_grid_sha256,
        "candidate_config_sha256": context.proof.candidate_config_sha256,
        "source_closure_sha256": context.evaluation_source_closure_sha256,
        "runtime_lock_sha256": context.runtime_lock_sha256,
        "delivery_months": list(context.delivery_months),
        "base_history_row_hashes": list(history_hashes),
        "base_retained_row_hashes": list(retained_hashes),
        "baseline_id": base_output.base_replay_module.BASELINE_ID,
        "core_output_schema": base_output.base_replay_module.BASE_REPLAY_OUTPUT_SCHEMA,
        "core_output_status": base_output.base_replay_module.BASE_REPLAY_STATUS,
        "expected_output": {
            "path": expected_path.name,
            "sha256": hashlib.sha256(expected_bytes).hexdigest(),
            "size_bytes": len(expected_bytes),
        },
    }
    output_manifest["base_input_set_sha256"] = tier2._sha256_json(
        {
            "schema_version": "tier2_monthly_eex_base_input_set.v1",
            "base_history_row_hashes": output_manifest["base_history_row_hashes"],
            "base_retained_row_hashes": output_manifest["base_retained_row_hashes"],
        }
    )
    _repin(output_manifest, "manifest_id")
    output_manifest_path = output_package / "base_output_manifest.v1.json"
    _canonical_write(output_manifest_path, output_manifest)
    model_private, model_public = _write_keypair(root, "model-execution")
    monkeypatch.setenv(base_output.MODEL_EXECUTION_PUBLIC_KEY_ENV, str(model_public))
    model_execution_path = root / "model-execution.json"
    output_timestamp_path = root / "model-output-time.json"
    _canonical_write(
        model_execution_path,
        _execution_attestation(
            output_manifest_path,
            model_private,
            executed_at="2026-01-02T08:00:02Z",
            role=base_output.BASE_OUTPUT_MANIFEST_ROLE,
            artifact_schema=base_output.BASE_OUTPUT_MANIFEST_SCHEMA,
        ),
    )
    _canonical_write(
        output_timestamp_path,
        _artifact_time_receipt(
            output_manifest_path,
            Path(governed_fold["tier2_time_private"]),
            role=base_output.BASE_OUTPUT_MANIFEST_ROLE,
            artifact_schema=base_output.BASE_OUTPUT_MANIFEST_SCHEMA,
            received_at="2026-01-02T08:00:03Z",
        ),
    )
    return {
        **paths,
        "base_output_manifest_path": output_manifest_path,
        "model_execution_path": model_execution_path,
        "output_timestamp_path": output_timestamp_path,
        "model_execution_private": model_private,
        "model_execution_public": model_public,
    }


@pytest.fixture
def governed_selection_input_set(
    governed_fold: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    root = tmp_path / "complete-selection-input-set"
    root.mkdir()
    selection_execution = _write_keypair(root, "selection-input-execution")
    journal_witness = _write_keypair(root, "journal-witness")
    monkeypatch.setenv(
        input_evidence.SELECTION_INPUT_EXECUTION_PUBLIC_KEY_ENV,
        str(selection_execution[1]),
    )
    monkeypatch.setenv(
        input_evidence.TIMESTAMP_JOURNAL_WITNESS_PUBLIC_KEY_ENV,
        str(journal_witness[1]),
    )
    plan = json.loads(Path(governed_fold["plan_path"]).read_text(encoding="utf-8"))
    catalog_path, history_path, catalog_root = _build_signed_eex_catalog(
        root,
        acquisition_private=Path(governed_fold["acquisition_private"]),
        source_time_private=Path(governed_fold["source-time_private"]),
        selection_folds=plan["selection_folds"],
    )
    source_closure_path = _write_source_closure(root)
    runtime_lock_path = _write_runtime_lock(root)
    plan["schema_version"] = tier2.SELECTION_INPUT_PLAN_SCHEMA
    plan["selection_input_resource_profile"] = dict(
        tier2.EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE
    )
    plan["selection_eex_root"] = catalog_root
    plan["evaluation_source_closure_sha256"] = hashlib.sha256(
        source_closure_path.read_bytes()
    ).hexdigest()
    plan["runtime_lock_sha256"] = hashlib.sha256(
        runtime_lock_path.read_bytes()
    ).hexdigest()
    _repin(plan, "plan_id")
    plan_path = root / "plan.json"
    governance_path = root / "plan-governance.json"
    timestamp_path = root / "plan-time.json"
    _canonical_write(plan_path, plan)
    _canonical_write(
        governance_path,
        sign_model_governance_artifact_receipt(
            plan_path,
            artifact_role="tier2_monthly_eex_evaluation_plan",
            authority_id="MODEL-RISK",
            issued_at_utc="2026-01-01T08:00:00Z",
            data_cutoff_utc="2025-12-31T23:59:59Z",
            private_key_path=Path(governed_fold["governance_private"]),
        ),
    )
    _canonical_write(
        timestamp_path,
        _artifact_time_receipt(
            plan_path,
            Path(governed_fold["tier2_time_private"]),
            role="tier2_monthly_eex_evaluation_plan",
            artifact_schema=tier2.SELECTION_INPUT_PLAN_SCHEMA,
            received_at="2026-01-01T08:00:01Z",
        ),
    )
    verified_plan = tier2.verify_governed_plan_artifact(
        plan_path,
        governance_receipt_path=governance_path,
        timestamp_receipt_path=timestamp_path,
    )
    frame, catalog, source_authorities = fold_evidence._verify_eex_paths(
        catalog_path, history_path
    )
    derived = selection_inputs._derive_selection_base_input_set(
        frame, plan=verified_plan
    )
    package = root / "selection-input-package"
    package.mkdir()
    inputs_path = package / "selection_base_inputs.v1.json"
    _canonical_write(inputs_path, derived.canonical_payload())
    inputs_bytes = inputs_path.read_bytes()
    manifest: dict[str, object] = {
        "schema_version": input_evidence.SELECTION_INPUT_MANIFEST_SCHEMA,
        "artifact_role": input_evidence.SELECTION_INPUT_MANIFEST_ROLE,
        "scope": input_evidence.SELECTION_INPUT_SCOPE,
        "status": input_evidence.SELECTION_INPUT_STATUS,
        "full_tier2_pfc_status": tier2.FULL_TIER2_STATUS,
        "data_lineage_verified": True,
        "configuration_neutral": True,
        "model_output_verified": False,
        "configuration_selection_verified": False,
        "target_non_observation_verified": False,
        "target_reveal_order_verified": False,
        "process_isolation_verified": False,
        "metrics_verified": False,
        "holdout_verified": False,
        "campaign_eligible": False,
        "production_approved": False,
        "peak_offpeak_verified": False,
        "plan_id": verified_plan.plan_id,
        "plan_artifact_sha256": verified_plan.artifact_sha256,
        "plan_governance_receipt_sha256": verified_plan.governance_receipt_sha256,
        "plan_timestamp_receipt_sha256": verified_plan.timestamp_receipt_sha256,
        "catalog_root": catalog_root,
        "frame_semantic_sha256": fold_evidence._frame_semantic_sha256(frame),
        "source_authorities": source_authorities,
        "evaluation_source_closure_sha256": plan[
            "evaluation_source_closure_sha256"
        ],
        "runtime_lock_sha256": plan["runtime_lock_sha256"],
        "resource_profile": dict(tier2.EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE),
        "fold_count": derived.fold_count,
        "ordered_fold_ids": list(derived.ordered_fold_ids),
        "input_set_id": derived.input_set_id,
        "inputs_artifact": {
            "path": inputs_path.name,
            "schema_version": selection_inputs.SELECTION_BASE_INPUT_SET_SCHEMA,
            "sha256": hashlib.sha256(inputs_bytes).hexdigest(),
            "size_bytes": len(inputs_bytes),
        },
    }
    _repin(manifest, "manifest_id")
    manifest_path = package / "selection_base_inputs_manifest.v1.json"
    _canonical_write(manifest_path, manifest)
    execution_path = root / "selection-input-execution.json"
    _canonical_write(
        execution_path,
        _execution_attestation(
            manifest_path,
            selection_execution[0],
            executed_at="2026-01-01T08:00:02Z",
            role=input_evidence.SELECTION_INPUT_MANIFEST_ROLE,
            artifact_schema=input_evidence.SELECTION_INPUT_MANIFEST_SCHEMA,
        ),
    )
    time_path = root / "selection-input-time.json"
    _canonical_write(
        time_path,
        _linked_timestamp_receipt(
            manifest_path,
            execution_path,
            timestamp_path,
            Path(governed_fold["tier2_time_private"]),
        ),
    )
    head_path = root / "selection-input-journal-head.json"
    _canonical_write(
        head_path,
        _journal_head(timestamp_path, time_path, journal_witness[0]),
    )
    monkeypatch.setenv(input_evidence.TIMESTAMP_JOURNAL_HEAD_PATH_ENV, str(head_path))
    head_payload = json.loads(head_path.read_text(encoding="utf-8"))
    monkeypatch.setenv(
        input_evidence.TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV,
        str(head_payload["current_head_id"]),
    )
    return {
        "plan_path": plan_path,
        "plan_governance_path": governance_path,
        "plan_timestamp_path": timestamp_path,
        "catalog_path": catalog_path,
        "history_path": history_path,
        "source_closure_path": source_closure_path,
        "runtime_lock_path": runtime_lock_path,
        "selection_input_manifest_path": manifest_path,
        "selection_input_execution_path": execution_path,
        "selection_input_time_path": time_path,
        "journal_head_path": head_path,
        "selection_input_execution_private": selection_execution[0],
        "journal_witness_private": journal_witness[0],
        "tier2_time_private": governed_fold["tier2_time_private"],
        "governance_private": governed_fold["governance_private"],
    }


@pytest.fixture
def governed_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    keys = {
        name: _write_keypair(tmp_path, name)
        for name in ("acquisition", "source-time", "governance", "tier2-time", "execution")
    }
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        str(keys["acquisition"][1]),
    )
    monkeypatch.setenv(
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
        str(keys["source-time"][1]),
    )
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_JOURNAL_ID", "source-journal")
    monkeypatch.setenv(
        MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
        str(keys["governance"][1]),
    )
    monkeypatch.setenv(tier2.TIMESTAMP_PUBLIC_KEY_ENV, str(keys["tier2-time"][1]))
    monkeypatch.setenv(tier2.TIMESTAMP_JOURNAL_ID_ENV, "tier2-journal")
    monkeypatch.setenv(
        tier2.EXECUTION_PUBLIC_KEY_ENV,
        str(keys["execution"][1]),
    )

    catalog_path, history_path, catalog_root = _build_signed_eex_catalog(
        tmp_path,
        acquisition_private=keys["acquisition"][0],
        source_time_private=keys["source-time"][0],
    )
    grid_path, config_hashes = _build_candidate_grid(tmp_path)
    plan = _build_plan(
        selection_eex_root=catalog_root,
        candidate_grid_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        candidate_config_hashes=config_hashes,
        source_closure_sha256="a" * 64,
        runtime_lock_sha256="b" * 64,
    )
    plan_path = tmp_path / "plan.json"
    _canonical_write(plan_path, plan)
    plan_governance = sign_model_governance_artifact_receipt(
        plan_path,
        artifact_role="tier2_monthly_eex_evaluation_plan",
        authority_id="MODEL-RISK",
        issued_at_utc="2026-01-01T08:00:00Z",
        data_cutoff_utc="2025-12-31T23:59:59Z",
        private_key_path=keys["governance"][0],
    )
    plan_governance_path = tmp_path / "plan-governance.json"
    plan_timestamp_path = tmp_path / "plan-time.json"
    _canonical_write(plan_governance_path, plan_governance)
    _canonical_write(
        plan_timestamp_path,
        _artifact_time_receipt(
            plan_path,
            keys["tier2-time"][0],
            role="tier2_monthly_eex_evaluation_plan",
            artifact_schema=tier2.PLAN_SCHEMA,
            received_at="2026-01-01T08:00:01Z",
        ),
    )

    fold_spec = next(
        fold
        for fold in plan["selection_folds"]
        if fold["fold_as_of_utc"] == "2025-03-15T08:00:00+00:00"
        and fold["tenor_horizon_bucket"] == "MONTH_H2"
    )
    manifest_path, rows_path, result_path = _build_fold_package(
        tmp_path,
        plan=plan,
        fold_spec=fold_spec,
        grid_path=grid_path,
        candidate_config_hash=config_hashes[0],
        catalog_path=catalog_path,
        history_path=history_path,
    )
    execution_path = tmp_path / "fold-execution.json"
    fold_timestamp_path = tmp_path / "fold-time.json"
    _canonical_write(
        execution_path,
        _execution_attestation(
            manifest_path,
            keys["execution"][0],
            executed_at="2026-01-02T08:00:00Z",
        ),
    )
    _canonical_write(
        fold_timestamp_path,
        _artifact_time_receipt(
            manifest_path,
            keys["tier2-time"][0],
            role=fold_evidence.FOLD_MANIFEST_ROLE,
            artifact_schema=fold_evidence.FOLD_MANIFEST_SCHEMA,
            received_at="2026-01-02T08:00:01Z",
        ),
    )
    return {
        "plan_path": plan_path,
        "plan_governance_path": plan_governance_path,
        "plan_timestamp_path": plan_timestamp_path,
        "grid_path": grid_path,
        "manifest_path": manifest_path,
        "rows_path": rows_path,
        "result_path": result_path,
        "execution_path": execution_path,
        "fold_timestamp_path": fold_timestamp_path,
        "catalog_path": catalog_path,
        "history_path": history_path,
        "fold_id": fold_spec["fold_id"],
        "catalog_id": catalog_root["catalog_id"],
        "execution_private": keys["execution"][0],
        "execution_public": keys["execution"][1],
        "governance_private": keys["governance"][0],
        "governance_public": keys["governance"][1],
        "tier2_time_private": keys["tier2-time"][0],
        "acquisition_private": keys["acquisition"][0],
        "acquisition_public": keys["acquisition"][1],
        "source-time_private": keys["source-time"][0],
        "source-time_public": keys["source-time"][1],
        "source_document_path": tmp_path / "sources" / "eex-2024-01-10.xlsx",
    }


def _verify(paths: MappingLike) -> fold_evidence.VerifiedSelectionFoldDataLineage:
    return fold_evidence.verify_signed_selection_fold_evidence(
        plan_path=paths["plan_path"],
        plan_governance_receipt_path=paths["plan_governance_path"],
        plan_timestamp_receipt_path=paths["plan_timestamp_path"],
        candidate_grid_path=paths["grid_path"],
        fold_manifest_path=paths["manifest_path"],
        execution_attestation_path=paths["execution_path"],
        fold_timestamp_receipt_path=paths["fold_timestamp_path"],
        eex_catalog_path=paths["catalog_path"],
        eex_history_path=paths["history_path"],
    )


def _verify_selection_inputs(
    paths: MappingLike,
) -> input_evidence.VerifiedSelectionBaseInputEvidence:
    return input_evidence.verify_signed_selection_base_inputs(
        plan_path=paths["plan_path"],
        plan_governance_receipt_path=paths["plan_governance_path"],
        plan_timestamp_receipt_path=paths["plan_timestamp_path"],
        eex_catalog_path=paths["catalog_path"],
        eex_history_path=paths["history_path"],
        selection_input_manifest_path=paths["selection_input_manifest_path"],
        execution_attestation_path=paths["selection_input_execution_path"],
        linked_timestamp_receipt_path=paths["selection_input_time_path"],
        journal_head_path=paths["journal_head_path"],
        source_closure_path=paths["source_closure_path"],
        runtime_lock_path=paths["runtime_lock_path"],
    )


def _resign_selection_input_chain(paths: MappingLike) -> None:
    manifest_path = Path(paths["selection_input_manifest_path"])
    execution_path = Path(paths["selection_input_execution_path"])
    time_path = Path(paths["selection_input_time_path"])
    head_path = Path(paths["journal_head_path"])
    _canonical_write(
        execution_path,
        _execution_attestation(
            manifest_path,
            Path(paths["selection_input_execution_private"]),
            executed_at="2026-01-01T08:00:02Z",
            role=input_evidence.SELECTION_INPUT_MANIFEST_ROLE,
            artifact_schema=input_evidence.SELECTION_INPUT_MANIFEST_SCHEMA,
        ),
    )
    _canonical_write(
        time_path,
        _linked_timestamp_receipt(
            manifest_path,
            execution_path,
            Path(paths["plan_timestamp_path"]),
            Path(paths["tier2_time_private"]),
        ),
    )
    _canonical_write(
        head_path,
        _journal_head(
            plan_receipt_path=Path(paths["plan_timestamp_path"]),
            receipt_path=time_path,
            private_path=Path(paths["journal_witness_private"]),
        ),
    )
    head = json.loads(head_path.read_text(encoding="utf-8"))
    os.environ[input_evidence.TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV] = str(
        head["current_head_id"]
    )


def _rebind_selection_input_payload(
    paths: MappingLike, payload: dict[str, object]
) -> None:
    manifest_path = Path(paths["selection_input_manifest_path"])
    payload_path = manifest_path.parent / "selection_base_inputs.v1.json"
    _canonical_write(payload_path, payload)
    payload_bytes = payload_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fold_count"] = payload["fold_count"]
    manifest["ordered_fold_ids"] = payload["ordered_fold_ids"]
    manifest["input_set_id"] = payload["input_set_id"]
    manifest["inputs_artifact"]["sha256"] = hashlib.sha256(payload_bytes).hexdigest()
    manifest["inputs_artifact"]["size_bytes"] = len(payload_bytes)
    _repin(manifest, "manifest_id")
    _canonical_write(manifest_path, manifest)
    _resign_selection_input_chain(paths)


def _append_witnessed_journal_entry(paths: MappingLike) -> None:
    head_path = Path(paths["journal_head_path"])
    payload = json.loads(head_path.read_text(encoding="utf-8"))
    payload.pop("witness_attestation")
    payload.pop("journal_head_id")
    previous = payload["entries"][-1]
    entry: dict[str, object] = {
        "head_sequence": int(previous["head_sequence"]) + 1,
        "receipt_id": "a" * 64,
        "receipt_sha256": "b" * 64,
        "receipt_journal_sequence": int(previous["receipt_journal_sequence"]) + 1,
        "previous_head_id": previous["head_id"],
    }
    entry["head_id"] = tier2._sha256_json(entry)
    payload["entries"].append(entry)
    payload["current_head_sequence"] = entry["head_sequence"]
    payload["current_head_id"] = entry["head_id"]
    payload["journal_head_id"] = tier2._sha256_json(payload)
    private = serialization.load_pem_private_key(
        Path(paths["journal_witness_private"]).read_bytes(), password=None
    )
    _canonical_write(
        head_path,
        _sign_payload(
            payload,
            private,
            signature_field="witness_attestation",
            id_payload_sha=True,
        ),
    )
    os.environ[input_evidence.TIMESTAMP_JOURNAL_CURRENT_HEAD_ID_ENV] = str(
        entry["head_id"]
    )


def _verify_context(
    paths: MappingLike,
) -> fold_evidence._VerifiedSelectionFoldReplayContext:
    return fold_evidence._verify_signed_selection_fold_evidence_context(
        plan_path=paths["plan_path"],
        plan_governance_receipt_path=paths["plan_governance_path"],
        plan_timestamp_receipt_path=paths["plan_timestamp_path"],
        candidate_grid_path=paths["grid_path"],
        fold_manifest_path=paths["manifest_path"],
        execution_attestation_path=paths["execution_path"],
        fold_timestamp_receipt_path=paths["fold_timestamp_path"],
        eex_catalog_path=paths["catalog_path"],
        eex_history_path=paths["history_path"],
    )


def _verify_base_output(
    paths: MappingLike,
) -> base_output.VerifiedSelectionFoldBaseModelOutput:
    return base_output.verify_signed_selection_fold_base_model_output(
        plan_path=paths["plan_path"],
        plan_governance_receipt_path=paths["plan_governance_path"],
        plan_timestamp_receipt_path=paths["plan_timestamp_path"],
        candidate_grid_path=paths["grid_path"],
        fold_manifest_path=paths["manifest_path"],
        fold_execution_attestation_path=paths["execution_path"],
        fold_timestamp_receipt_path=paths["fold_timestamp_path"],
        eex_catalog_path=paths["catalog_path"],
        eex_history_path=paths["history_path"],
        base_output_manifest_path=paths["base_output_manifest_path"],
        model_execution_attestation_path=paths["model_execution_path"],
        base_output_timestamp_receipt_path=paths["output_timestamp_path"],
        source_closure_path=paths["source_closure_path"],
        runtime_lock_path=paths["runtime_lock_path"],
    )


MappingLike = dict[str, object]


def _build_candidate_grid(tmp_path: Path) -> tuple[Path, list[str]]:
    configs = [
        _base_candidate_config(lambda_smooth_month=value)
        for value in (1.0, 2.0)
    ]
    entries = sorted(
        (
            {"config_sha256": tier2._sha256_json(config), "config": config}
            for config in configs
        ),
        key=lambda entry: entry["config_sha256"],
    )
    grid: dict[str, object] = {
        "schema_version": fold_evidence.CANDIDATE_GRID_SCHEMA,
        "market": "CH",
        "timezone": "Europe/Zurich",
        "configs": entries,
    }
    grid["grid_id"] = tier2._sha256_json(grid)
    path = tmp_path / "candidate-grid.json"
    _canonical_write(path, grid)
    return path, [str(entry["config_sha256"]) for entry in entries]


def _base_candidate_config(*, lambda_smooth_month: float) -> dict[str, object]:
    return {
        "schema_version": BASE_REPLAY_CONFIG_SCHEMA,
        "monthly_curve": {
            "lambda_prior": 1e-6,
            "lambda_smooth_month": lambda_smooth_month,
            "lambda_smooth_yoy": 0.25,
            "lambda_shape": 1.0,
            "neighbor_shrinkage": 0.0,
            "robust_panel_quantile": 0.5,
            "min_history_snapshots": 24,
            "max_prior_residual_eur_mwh": None,
            "constraint_tolerance": 1e-9,
            "quote_conflict_tolerance": 0.01,
            "stationarity_tolerance": 1e-7,
        },
    }


def _write_source_closure(tmp_path: Path) -> Path:
    entries: list[dict[str, object]] = []
    for role, path in sorted(base_output._expected_source_paths().items()):
        snapshot = fold_evidence._file_snapshot(path)
        entries.append(
            {
                "role": role,
                "path": str(path),
                "sha256": snapshot[-1],
                "size_bytes": snapshot[2],
                "loaded_code_sha256": base_output._loaded_module_code_sha256(
                    base_output._expected_source_modules()[role]
                ),
            }
        )
    payload: dict[str, object] = {
        "schema_version": base_output.SOURCE_CLOSURE_SCHEMA,
        "files": entries,
    }
    _repin(payload, "closure_id")
    path = tmp_path / "source-closure.json"
    _canonical_write(path, payload)
    return path


def _write_runtime_lock(tmp_path: Path) -> Path:
    path = tmp_path / "runtime-lock.json"
    _canonical_write(path, base_output._current_runtime_payload())
    return path


def _build_signed_eex_catalog(
    tmp_path: Path,
    *,
    acquisition_private: Path,
    source_time_private: Path,
    rich_history: bool = False,
    selection_folds: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    source_private = serialization.load_pem_private_key(
        source_time_private.read_bytes(), password=None
    )
    sources = tmp_path / "sources"
    parser_dir = tmp_path / "parser"
    sources.mkdir()
    parser_dir.mkdir()
    parser_path = parser_dir / "ingest_forwards.py"
    parser_path.write_bytes(Path(ingest_forwards.__file__).read_bytes())
    parser_hash = hashlib.sha256(parser_path.read_bytes()).hexdigest()
    specs: list[tuple[str, str, list[tuple[str, float]]]] = [
        (
            "2024-01-10",
            "2024-01-10T08:00:00Z",
            [("M02_2027_BASE", 70.0)],
        ),
        (
            "2025-03-15",
            "2025-03-15T08:00:00Z",
            [
                ("M01_2027_BASE", 100.0),
                ("M02_2027_BASE", 80.0),
                ("Q01_2027_BASE", 90.0),
            ],
        ),
        (
            "2025-04-15",
            "2025-04-15T08:00:00Z",
            [("M01_2027_BASE", 101.0)],
        ),
    ]
    if rich_history:
        specs = []
        for index, period in enumerate(
            pd.period_range("2023-01", "2024-12", freq="M")
        ):
            date = f"{period.year}-{period.month:02d}-10"
            level = 80.0 + 0.05 * index
            products = [("Y01_2027_BASE", level)] + [
                (f"M{month:02d}_2027_BASE", level + float(month - 6.5))
                for month in range(1, 13)
            ]
            specs.append((date, f"{date}T08:00:00Z", products))
        specs.extend(
            [
                (
                    "2025-03-15",
                    "2025-03-15T08:00:00Z",
                    [
                        ("M01_2027_BASE", 100.0),
                        ("M02_2027_BASE", 80.0),
                        ("Q01_2027_BASE", 90.0),
                    ],
                ),
                (
                    "2025-04-15",
                    "2025-04-15T08:00:00Z",
                    [("M01_2027_BASE", 101.0)],
                ),
            ]
        )
    if selection_folds is not None:
        by_origin: dict[str, set[str]] = {}
        for fold in selection_folds:
            origin_date = str(fold["fold_as_of_utc"])[:10]
            by_origin.setdefault(origin_date, set()).add(str(fold["target_product"]))
        specs = []
        for origin_index, (date, targets) in enumerate(sorted(by_origin.items())):
            products: list[tuple[str, float]] = []
            for product_index, product in enumerate(sorted(targets)):
                if "-Q" in product:
                    year, quarter = product.split("-Q", maxsplit=1)
                    code = f"Q{int(quarter):02d}_{year}_BASE"
                else:
                    year, month = product.split("-", maxsplit=1)
                    code = f"M{int(month):02d}_{year}_BASE"
                products.append(
                    (code, 60.0 + origin_index * 0.1 + product_index * 0.01)
                )
            specs.append((date, f"{date}T08:00:00Z", products))
    all_rows: list[dict[str, object]] = []
    documents: list[dict[str, object]] = []
    previous_receipt_id = ""
    for sequence, (date, available, products) in enumerate(specs, start=1):
        source = sources / f"eex-{date}.xlsx"
        workbook = [
            [None, *[code for code, _ in products]],
            [None, *[f"ISIN-{index}" for index in range(len(products))]],
            ["Date", *([None] * len(products))],
            [pd.Timestamp(date).strftime("%d.%m.%Y"), *[price for _, price in products]],
        ]
        with pd.ExcelWriter(source, engine="openpyxl") as writer:
            pd.DataFrame(workbook).to_excel(
                writer, sheet_name="CH", index=False, header=False
            )
        payload = source.read_bytes()
        source_hash = hashlib.sha256(payload).hexdigest()
        prices, parsed_date, metadata = ingest_forwards.load_base_prices_from_eex_report_bytes(
            payload,
            market="CH",
            source_label=source.name,
            return_snapshot_metadata=True,
        )
        assert pd.Timestamp(parsed_date).date().isoformat() == date
        parser_config = {
            "parser_version": "ingest_forwards.v1",
            "selection_mode": "latest_available",
            "markets": {"CH": date},
        }
        parser_config_hash = _canonical_sha(parser_config)
        receipt_unsigned: dict[str, object] = {
            "schema_version": "eex_source_receipt.v1",
            "source_document_id": source.name,
            "source_document_sha256": source_hash,
            "size_bytes": len(payload),
            "received_at_utc": pd.Timestamp(available).isoformat(),
            "journal_id": "source-journal",
            "journal_sequence": sequence,
            "previous_receipt_id": previous_receipt_id,
        }
        receipt_unsigned["receipt_id"] = _canonical_sha(receipt_unsigned)
        receipt = _sign_payload(
            receipt_unsigned,
            source_private,
            signature_field="trusted_time_attestation",
            id_payload_sha=True,
        )
        previous_receipt_id = str(receipt_unsigned["receipt_id"])
        snapshot_rows: list[dict[str, object]] = []
        for price_key, price in prices.items():
            load_type = "BASE"
            product = str(price_key)
            if product.endswith("-Peak"):
                load_type = "PEAK"
                product = product.removesuffix("-Peak")
            elif product.endswith("-Offpeak"):
                load_type = "OFFPEAK"
                product = product.removesuffix("-Offpeak")
            lineage = metadata["quote_lineage"][price_key]
            row: dict[str, object] = {
                "date": pd.Timestamp(date),
                "observed_at": pd.Timestamp(available),
                "available_at": pd.Timestamp(available),
                "acquisition_id": receipt_unsigned["receipt_id"],
                "product": product,
                "load_type": load_type,
                "product_type": (
                    "QUARTER" if "-Q" in product else "MONTH" if len(product) == 7 else "CAL"
                ),
                "price": float(price),
                "unit": "EUR/MWH",
                "market": "CH",
                "source": "EEX",
                "source_document_id": source.name,
                "source_document_sha256": source_hash,
                "source_sheet": "CH",
                "source_row_index": int(lineage["source_row_index"]),
                "source_column_index": int(lineage["source_column_index"]),
                "source_product_code": str(lineage["source_product_code"]),
                "parser_version": "ingest_forwards.v1",
                "parser_code_sha256": parser_hash,
                "parser_config_sha256": parser_config_hash,
                "revision_sequence": 1,
                "supersedes_quote_id": "",
                "revision_timestamp": pd.Timestamp(available),
                "ingestion_run_id": f"ingest-{date}",
                "schema_version": EEX_HISTORICAL_QUOTE_SCHEMA,
            }
            _repin_row(row)
            snapshot_rows.append(row)
        all_rows.extend(snapshot_rows)
        documents.append(
            {
                "snapshot_ids": sorted({str(row["snapshot_id"]) for row in snapshot_rows}),
                "acquisition_id": receipt_unsigned["receipt_id"],
                "observed_at": pd.Timestamp(available).isoformat(),
                "available_at": pd.Timestamp(available).isoformat(),
                "source_document_id": source.name,
                "path": f"sources/{source.name}",
                "sha256": source_hash,
                "size_bytes": len(payload),
                "parser_code_path": "parser/ingest_forwards.py",
                "parser_code_sha256": parser_hash,
                "parser_config": parser_config,
                "parser_config_sha256": parser_config_hash,
                "trusted_time_receipt": receipt,
            }
        )
    history_path = tmp_path / "history.parquet"
    pd.DataFrame(all_rows).to_parquet(history_path, index=False)
    history_bytes = history_path.read_bytes()
    latest_available = max(pd.Timestamp(available) for _, available, _ in specs)
    unsigned: dict[str, object] = {
        "schema_version": EEX_HISTORICAL_CATALOG_SCHEMA,
        "created_at_utc": (latest_available + pd.Timedelta(hours=1)).isoformat(),
        "data_cutoff_utc": latest_available.isoformat(),
        "calibration_eligible": True,
        "source_class": "IMMUTABLE_DAILY_EEX_XLSX",
        "revision_policy": "APPEND_ONLY_PRESERVE_ALL_REVISIONS",
        "ompex_used": False,
        "history_parquet": {
            "path": "history.parquet",
            "sha256": hashlib.sha256(history_bytes).hexdigest(),
            "size_bytes": len(history_bytes),
            "row_count": len(all_rows),
        },
        "source_documents": documents,
    }
    unsigned["catalog_id"] = _canonical_sha(unsigned)
    catalog = sign_acquisition_contract(unsigned, private_key_path=acquisition_private)
    catalog_path = tmp_path / "catalog.json"
    _canonical_write(catalog_path, catalog)
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    root: dict[str, object] = {
        "catalog_id": unsigned["catalog_id"],
        "catalog_sha256": catalog_sha,
        "history_sha256": hashlib.sha256(history_bytes).hexdigest(),
    }
    root["root_sha256"] = tier2._sha256_json(
        {"schema_version": "tier2_monthly_eex_catalog_root.v1", **root}
    )
    return catalog_path, history_path, root


def _build_fold_package(
    tmp_path: Path,
    *,
    plan: dict[str, object],
    fold_spec: dict[str, object],
    grid_path: Path,
    candidate_config_hash: str,
    catalog_path: Path,
    history_path: Path,
) -> tuple[Path, Path, Path]:
    frame, evidence, authorities = fold_evidence._verify_eex_paths(
        catalog_path, history_path
    )
    fake_plan = SimpleNamespace(
        trusted_time_utc=pd.Timestamp("2026-01-01T08:00:01Z"), payload=plan
    )
    roles, target, retained, delivery_months = fold_evidence._derive_fold_rows(
        frame, plan=fake_plan, fold_spec=fold_spec
    )
    frame_hash = fold_evidence._frame_semantic_sha256(frame)
    rows = {
        "schema_version": fold_evidence.FOLD_ROWS_SCHEMA,
        "frame_semantic_sha256": frame_hash,
        "roles": roles,
        "permitted_candidate_input_row_hashes": [
            *roles["fold_historical_context"]["row_hashes"],
            *roles["fold_evaluation_snapshot_retained"]["row_hashes"],
        ],
    }
    candidate_segments = _segments(delivery_months, target_price=95.0)
    baseline_segments = _segments(delivery_months, target_price=105.0)
    target_constraint = ProductConstraint(str(target["product"]), str(target["load_type"]))
    retained_constraints = [
        ProductConstraint(str(row.product), str(row.load_type))
        for row in retained.itertuples(index=False)
    ]
    assessment = tier2.assess_target_identifiability(
        target=target_constraint, retained=retained_constraints
    )
    target_vector = product_constraint_vector(
        target_constraint, delivery_months=delivery_months
    )
    retained_matrix = [
        product_constraint_vector(item, delivery_months=delivery_months).tolist()
        for item in retained_constraints
    ]
    candidate_values = _segment_vector(candidate_segments)
    baseline_values = _segment_vector(baseline_segments)
    candidate_prediction = float(np.dot(target_vector, candidate_values))
    baseline_prediction = float(np.dot(target_vector, baseline_values))
    target_price = float(target["price"])
    candidate_error = candidate_prediction - target_price
    baseline_error = baseline_prediction - target_price
    metrics = {
        "target_quote_id": str(target["quote_id"]),
        "target_row_hash": str(target["row_hash"]),
        "target_product": str(target["product"]),
        "target_load_type": str(target["load_type"]),
        "target_price": target_price,
        "target_weight": float(fold_evidence._product_delivery_hours(target_constraint)),
        "candidate_prediction": candidate_prediction,
        "baseline_prediction": baseline_prediction,
        "candidate_signed_error": candidate_error,
        "baseline_signed_error": baseline_error,
        "candidate_abs_error": abs(candidate_error),
        "baseline_abs_error": abs(baseline_error),
        "candidate_squared_error": candidate_error**2,
        "baseline_squared_error": baseline_error**2,
        "candidate_repricing_max_abs_error": fold_evidence._repricing_max(
            retained, candidate_values, delivery_months=delivery_months
        ),
        "baseline_repricing_max_abs_error": fold_evidence._repricing_max(
            retained, baseline_values, delivery_months=delivery_months
        ),
        "candidate_conservation_max_abs_error": 0.0,
        "baseline_conservation_max_abs_error": 0.0,
        "target_vector_sha256": tier2._sha256_json(target_vector.tolist()),
        "retained_matrix_sha256": tier2._sha256_json(retained_matrix),
        "identifiability_status": assessment.status,
        "retained_rank": assessment.retained_rank,
        "augmented_rank": assessment.augmented_rank,
        "retained_condition_number": assessment.retained_condition_number,
        "augmented_condition_number": assessment.augmented_condition_number,
    }
    result = {
        "schema_version": fold_evidence.FOLD_RESULT_SCHEMA,
        "status": fold_evidence.FOLD_STATUS,
        "data_lineage_verified": True,
        "model_output_verified": False,
        "campaign_eligible": False,
        "production_approved": False,
        "candidate_config_sha256": candidate_config_hash,
        "baseline_config_sha256": tier2._sha256_json(plan["baseline"]),
        "delivery_months": list(delivery_months),
        "candidate_segments": candidate_segments,
        "baseline_segments": baseline_segments,
        "reported_metrics": metrics,
    }
    package = tmp_path / "fold-package"
    package.mkdir()
    rows_path = package / "fold_rows.v1.json"
    result_path = package / "fold_result.v1.json"
    _canonical_write(rows_path, rows)
    _canonical_write(result_path, result)
    manifest: dict[str, object] = {
        "schema_version": fold_evidence.FOLD_MANIFEST_SCHEMA,
        "scope": tier2.SCOPE,
        "full_tier2_pfc_status": tier2.FULL_TIER2_STATUS,
        "data_lineage_verified": True,
        "model_output_verified": False,
        "campaign_eligible": False,
        "production_approved": False,
        "status": fold_evidence.FOLD_STATUS,
        "plan_id": plan["plan_id"],
        "plan_artifact_sha256": "",
        "fold_id": fold_spec["fold_id"],
        "fold_spec_hash": fold_spec["fold_spec_hash"],
        "campaign_type": "SELECTION",
        "origin_id": fold_spec["origin_id"],
        "fold_as_of_utc": fold_spec["fold_as_of_utc"],
        "candidate_grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "candidate_config_sha256": candidate_config_hash,
        "evaluation_source_closure_sha256": plan["evaluation_source_closure_sha256"],
        "runtime_lock_sha256": plan["runtime_lock_sha256"],
        "catalog_root": plan["selection_eex_root"],
        "frame_semantic_sha256": frame_hash,
        "source_authorities": authorities,
        "artifacts": [
            _artifact_entry("fold_rows", rows_path, fold_evidence.FOLD_ROWS_SCHEMA),
            _artifact_entry("fold_result", result_path, fold_evidence.FOLD_RESULT_SCHEMA),
        ],
    }
    manifest["plan_artifact_sha256"] = hashlib.sha256(
        tier2._canonical_json_bytes(plan)
    ).hexdigest()
    _repin(manifest, "evidence_id")
    manifest_path = package / "fold_manifest.v1.json"
    _canonical_write(manifest_path, manifest)
    return manifest_path, rows_path, result_path


def _segments(delivery_months: tuple[str, ...], *, target_price: float) -> list[dict[str, object]]:
    rows = []
    for month in delivery_months:
        price = target_price if month == "2027-01" else 80.0
        rows.append(
            {
                "month": month,
                "peak_price": price,
                "offpeak_price": price,
                "base_price": price,
            }
        )
    return rows


def _segment_vector(rows: list[dict[str, object]]) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        values.extend([float(row["peak_price"]), float(row["offpeak_price"])])
    return np.asarray(values, dtype=float)


def _build_plan(
    *,
    selection_eex_root: dict[str, object],
    candidate_grid_sha256: str,
    candidate_config_hashes: list[str],
    source_closure_sha256: str,
    runtime_lock_sha256: str,
) -> dict[str, object]:
    plan: dict[str, object] = {
        "schema_version": tier2.PLAN_SCHEMA,
        "scope": tier2.SCOPE,
        "full_tier2_pfc_status": tier2.FULL_TIER2_STATUS,
        "production_approved": False,
        "market": "CH",
        "timezone": "Europe/Zurich",
        "estimand": {
            "unit": "ONE_ECONOMIC_SAME_VINTAGE_EEX_QUOTE",
            "target_removed_from_all_inputs": True,
            "algebraically_revealing_quotes_removed": True,
            "retained_hard_quotes_repriced": True,
            "later_quotes_namespace": "DIAGNOSTIC_EXCLUDED",
            "allowed_sources": ["EEX"],
            "forbidden_sources": ["OMPEX", "PROXY", "BENCHMARK"],
        },
        "baseline": dict(tier2._EXPECTED_BASELINE),
        "secondary_challengers": ["parent_flat.v1", "last_visible_quote.v1"],
        "coverage": dict(tier2._EXPECTED_COVERAGE),
        "performance": dict(tier2._EXPECTED_PERFORMANCE),
        "bootstrap": dict(tier2._EXPECTED_BOOTSTRAP),
        "identifiability": dict(tier2._EXPECTED_IDENTIFIABILITY),
        "status_precedence": ["ERROR", "FAIL", "UNSUPPORTED", tier2.PASS_STATUS],
        "candidate_grid_sha256": candidate_grid_sha256,
        "candidate_grid_contract": dict(tier2._EXPECTED_CANDIDATE_GRID_CONTRACT),
        "candidate_config_sha256_allowlist": candidate_config_hashes,
        "selection_rule": "MIN_SELECTION_MAE_THEN_RMSE_LEXICOGRAPHIC_FIXED_GRID",
        "selection_eex_root": selection_eex_root,
        "fold_source_policy": dict(tier2._EXPECTED_FOLD_SOURCE_POLICY),
        "fold_context_policy": dict(tier2._EXPECTED_FOLD_CONTEXT_POLICY),
        "fold_weight_policy": dict(tier2._EXPECTED_FOLD_WEIGHT_POLICY),
        "fold_claim_policy": dict(tier2._EXPECTED_FOLD_CLAIM_POLICY),
        "evaluation_source_closure_sha256": source_closure_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "mandatory_tenor_horizon_buckets": list(tier2.MANDATORY_TENOR_HORIZON_BUCKETS),
        "selection_folds": _campaign_folds("SELECTION", origins=36, first_year=2023),
        "holdout_folds": _campaign_folds("HOLDOUT", origins=24, first_year=2027),
    }
    _repin(plan, "plan_id")
    return plan


def _campaign_folds(campaign: str, *, origins: int, first_year: int) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    offset = 0 if campaign == "SELECTION" else 1_000
    for index in range(1, origins + 1):
        year = first_year + (index - 1) // 12
        month = (index - 1) % 12 + 1
        date = f"{year}-{month:02d}-15"
        for bucket in tier2.MANDATORY_TENOR_HORIZON_BUCKETS:
            fold = _fold_spec(campaign, offset + index, date, bucket)
            if fold is not None:
                folds.append(fold)
    return folds


def _fold_spec(
    campaign: str, number: int, date: str, bucket: str
) -> dict[str, object] | None:
    tenor, horizon = bucket.split("_", maxsplit=1)
    year = int(date[:4])
    month = int(date[5:7])
    horizon_years = {"H0": 0, "H1": 1, "H2": 2, "H3_PLUS": 3}[horizon]
    delivery_year = year + horizon_years
    if horizon == "H0" and tenor == "MONTH":
        if month == 12:
            return None
        product = f"{delivery_year}-{month + 1:02d}"
    elif horizon == "H0":
        quarter = (month - 1) // 3 + 1
        if quarter == 4:
            return None
        product = f"{delivery_year}-Q{quarter + 1}"
    else:
        product = f"{delivery_year}-01" if tenor == "MONTH" else f"{delivery_year}-Q1"
    fold_as_of = f"{date}T08:00:00+00:00"
    fold: dict[str, object] = {
        "fold_id": f"{campaign.lower()}-{number:03d}-{bucket}",
        "campaign_type": campaign,
        "origin_id": tier2._planned_origin_id(campaign, fold_as_of),
        "fold_as_of_utc": fold_as_of,
        "tenor_horizon_bucket": bucket,
        "target_product": product,
        "target_load_type": "BASE",
        "target_rule": "MASK_ONE_IDENTIFIABLE_SAME_VINTAGE_EEX_QUOTE",
        "historical_context_rule": "EXPANDING_PIT_AVAILABLE_AT_LE_FOLD_AS_OF_NO_RETUNING",
    }
    _repin(fold, "fold_spec_hash")
    return fold


def _artifact_entry(role: str, path: Path, schema: str) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "role": role,
        "path": path.name,
        "schema_version": schema,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _rebind_package_and_receipts(paths: dict[str, object]) -> None:
    manifest_path = Path(paths["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = manifest_path.parent
    manifest["artifacts"] = [
        _artifact_entry("fold_rows", package / "fold_rows.v1.json", fold_evidence.FOLD_ROWS_SCHEMA),
        _artifact_entry("fold_result", package / "fold_result.v1.json", fold_evidence.FOLD_RESULT_SCHEMA),
    ]
    _repin(manifest, "evidence_id")
    _canonical_write(manifest_path, manifest)
    _resign_fold_receipts(paths)


def _resign_fold_receipts(paths: dict[str, object]) -> None:
    _canonical_write(
        Path(paths["execution_path"]),
        _execution_attestation(
            Path(paths["manifest_path"]),
            Path(paths["execution_private"]),
            executed_at="2026-01-02T08:00:00Z",
        ),
    )
    _canonical_write(
        Path(paths["fold_timestamp_path"]),
        _artifact_time_receipt(
            Path(paths["manifest_path"]),
            Path(paths["tier2_time_private"]),
            role=fold_evidence.FOLD_MANIFEST_ROLE,
            artifact_schema=fold_evidence.FOLD_MANIFEST_SCHEMA,
            received_at="2026-01-02T08:00:01Z",
        ),
    )


def _resign_base_output_receipts(paths: dict[str, object]) -> None:
    manifest = Path(paths["base_output_manifest_path"])
    _canonical_write(
        Path(paths["model_execution_path"]),
        _execution_attestation(
            manifest,
            Path(paths["model_execution_private"]),
            executed_at="2026-01-02T08:00:02Z",
            role=base_output.BASE_OUTPUT_MANIFEST_ROLE,
            artifact_schema=base_output.BASE_OUTPUT_MANIFEST_SCHEMA,
        ),
    )
    _canonical_write(
        Path(paths["output_timestamp_path"]),
        _artifact_time_receipt(
            manifest,
            Path(paths["tier2_time_private"]),
            role=base_output.BASE_OUTPUT_MANIFEST_ROLE,
            artifact_schema=base_output.BASE_OUTPUT_MANIFEST_SCHEMA,
            received_at="2026-01-02T08:00:03Z",
        ),
    )


def _artifact_time_receipt(
    artifact: Path,
    private_path: Path,
    *,
    role: str,
    artifact_schema: str,
    received_at: str,
) -> dict[str, object]:
    private = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    raw = artifact.read_bytes()
    unsigned: dict[str, object] = {
        "schema_version": tier2.TIMESTAMP_RECEIPT_SCHEMA,
        "artifact_role": role,
        "artifact_schema_version": artifact_schema,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_size_bytes": len(raw),
        "authority_id": "TRUSTED-TIME",
        "received_at_utc": received_at,
        "journal_id": "tier2-journal",
        "journal_sequence": 1,
        "previous_receipt_id": "",
    }
    unsigned["receipt_id"] = tier2._sha256_json(unsigned)
    return _sign_payload(
        unsigned,
        private,
        signature_field="trusted_time_attestation",
        id_payload_sha=True,
    )


def _linked_timestamp_receipt(
    artifact: Path,
    execution_path: Path,
    plan_timestamp_path: Path,
    private_path: Path,
) -> dict[str, object]:
    private = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    artifact_bytes = artifact.read_bytes()
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_bytes = execution_path.read_bytes()
    plan_time = json.loads(plan_timestamp_path.read_text(encoding="utf-8"))
    unsigned: dict[str, object] = {
        "schema_version": input_evidence.LINKED_TIMESTAMP_RECEIPT_SCHEMA,
        "artifact_role": input_evidence.SELECTION_INPUT_MANIFEST_ROLE,
        "artifact_schema_version": input_evidence.SELECTION_INPUT_MANIFEST_SCHEMA,
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "artifact_size_bytes": len(artifact_bytes),
        "authority_id": "TRUSTED-TIME",
        "received_at_utc": "2026-01-01T08:00:03Z",
        "journal_id": plan_time["journal_id"],
        "journal_sequence": int(plan_time["journal_sequence"]) + 1,
        "previous_receipt_id": plan_time["receipt_id"],
        "execution_attestation_id": execution["attestation_id"],
        "execution_attestation_sha256": hashlib.sha256(execution_bytes).hexdigest(),
        "plan_timestamp_receipt_sha256": hashlib.sha256(
            plan_timestamp_path.read_bytes()
        ).hexdigest(),
    }
    unsigned["receipt_id"] = tier2._sha256_json(unsigned)
    return _sign_payload(
        unsigned,
        private,
        signature_field="trusted_time_attestation",
        id_payload_sha=True,
    )


def _journal_head(
    plan_receipt_path: Path,
    receipt_path: Path,
    private_path: Path,
) -> dict[str, object]:
    private = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    receipt_paths = (plan_receipt_path, receipt_path)
    entries: list[dict[str, object]] = []
    previous_head_id = ""
    for index, current_path in enumerate(receipt_paths):
        receipt = json.loads(current_path.read_text(encoding="utf-8"))
        entry: dict[str, object] = {
            "head_sequence": index,
            "receipt_id": receipt["receipt_id"],
            "receipt_sha256": hashlib.sha256(current_path.read_bytes()).hexdigest(),
            "receipt_journal_sequence": receipt["journal_sequence"],
            "previous_head_id": previous_head_id,
        }
        entry["head_id"] = tier2._sha256_json(entry)
        previous_head_id = str(entry["head_id"])
        entries.append(entry)
    unsigned: dict[str, object] = {
        "schema_version": input_evidence.JOURNAL_HEAD_SCHEMA,
        "journal_id": json.loads(
            receipt_path.read_text(encoding="utf-8")
        )["journal_id"],
        "entries": entries,
        "current_head_sequence": len(entries) - 1,
        "current_head_id": previous_head_id,
        "witness_authority_id": "JOURNAL-WITNESS",
        "witnessed_at_utc": "2026-01-01T08:00:04Z",
    }
    unsigned["journal_head_id"] = tier2._sha256_json(unsigned)
    return _sign_payload(
        unsigned,
        private,
        signature_field="witness_attestation",
        id_payload_sha=True,
    )


def _execution_attestation(
    artifact: Path,
    private_path: Path,
    *,
    executed_at: str,
    role: str = fold_evidence.FOLD_MANIFEST_ROLE,
    artifact_schema: str = fold_evidence.FOLD_MANIFEST_SCHEMA,
) -> dict[str, object]:
    private = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    raw = artifact.read_bytes()
    unsigned: dict[str, object] = {
        "schema_version": tier2.EXECUTION_ATTESTATION_SCHEMA,
        "artifact_role": role,
        "artifact_schema_version": artifact_schema,
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_size_bytes": len(raw),
        "authority_id": "EXECUTION",
        "executed_at_utc": executed_at,
    }
    unsigned["attestation_id"] = tier2._sha256_json(unsigned)
    return _sign_payload(
        unsigned,
        private,
        signature_field="execution_attestation",
        id_payload_sha=True,
    )


def _sign_payload(
    unsigned: dict[str, object],
    private: Ed25519PrivateKey,
    *,
    signature_field: str,
    id_payload_sha: bool,
) -> dict[str, object]:
    payload = tier2._canonical_json_bytes(unsigned)
    public_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    block = {
        "algorithm": "Ed25519",
        "key_id": hashlib.sha256(public_raw).hexdigest(),
        "value_base64": base64.b64encode(private.sign(payload)).decode("ascii"),
    }
    if id_payload_sha:
        block["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    signed = dict(unsigned)
    signed[signature_field] = block
    return signed


def _repin_row(row: dict[str, object]) -> None:
    row["snapshot_id"] = historical_vintage_snapshot_id(row)
    row["revision_id"] = historical_vintage_revision_id(row)
    row["row_hash"] = historical_vintage_row_hash(row)
    row["quote_id"] = row["row_hash"]


def _repin(payload: dict[str, object], field: str) -> None:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    payload[field] = tier2._sha256_json(unsigned)


def _canonical_write(path: Path, payload: object) -> None:
    path.write_bytes(tier2._canonical_json_bytes(payload))


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(tier2._canonical_json_bytes(payload)).hexdigest()


def _write_keypair(tmp_path: Path, stem: str) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / f"{stem}.private.pem"
    public_path = tmp_path / f"{stem}.public.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path
