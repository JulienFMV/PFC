from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.data.ingest_forwards as ingest_forwards
import pfc_shaping.pipeline.atomic_promotion as atomic_promotion
import scripts.check_monthly_curve_promotion_from_manifests as capstone
from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    build_monthly_constraint_system,
)
from pfc_shaping.data import ingest_entso, ingest_epex, ingest_hydro
from pfc_shaping.data.acquisition_contract import (
    TRUSTED_TIME_RECEIPT_SCHEMA,
    governed_snapshot_bundle_root_sha256,
)
from pfc_shaping.data.acquisition_signing import (
    sign_acquisition_contract,
    sign_eex_trusted_time_receipt,
    sign_source_acquisition_receipt,
    sign_source_journal_checkpoint,
)
from pfc_shaping.data.eex_historical_vintage import (
    EEX_HISTORICAL_CATALOG_SCHEMA,
    EEX_HISTORICAL_QUOTE_SCHEMA,
    historical_vintage_revision_id,
    historical_vintage_row_hash,
    historical_vintage_snapshot_id,
    materialize_eex_forward_history_as_of,
)
from pfc_shaping.data.forward_proxy import ForwardEligibility, ForwardSnapshot
from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report_bytes
from pfc_shaping.data.lt_input_replay import (
    approved_parser_path_for_role,
    build_replay_config,
)
from pfc_shaping.data.lt_input_sources import dataframe_sha256
from pfc_shaping.data.snapshot_anchor_client import SnapshotAnchorClientError
from pfc_shaping.data.snapshot_anchor_signing import (
    sign_snapshot_anchor_head_observation,
    sign_snapshot_anchor_receipt,
)
from pfc_shaping.data.snapshot_publication_contract import (
    PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
    PUBLICATION_DOMAIN_ID_ENV,
    PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV,
    PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV,
    publication_domain_sha256,
    sign_snapshot_publication_intent,
)
from pfc_shaping.data.snapshot_publication_state import (
    canonical_json,
    external_head_challenge_sha256,
    pointer_mapping_from_external_receipt,
)
from pfc_shaping.pipeline.atomic_promotion import PromotionError, promote_candidate
from pfc_shaping.pipeline.candidate_evidence import (
    EVIDENCE_MANIFEST,
    GOVERNED_RUN_SPEC_SCHEMA,
    CandidateEvidenceError,
    EvidenceArtifactInput,
    _canonical_mapping_sha256,
    _run_spec_artifact_bindings,
    seal_candidate_evidence,
)
from pfc_shaping.pipeline.model_governance_contract import (
    sign_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    active_monthly_curve_config_payload,
    monthly_solver_settings,
)
from pfc_shaping.pipeline.quality_gate import validate_input_frame
from pfc_shaping.pipeline.release_request_contract import (
    ensure_workflow_domain,
    sign_release_request,
)
from pfc_shaping.version import __version__
from scripts.audit_ch_product_normalization import eex_peak_mask, run_audit
from scripts.check_monthly_curve_promotion_from_manifests import (
    _active_config_hash,
    _active_config_payload,
    main,
)


def test_load_mapping_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "ambiguous.json"
    artifact.write_text(
        '{"production_approved":false,"production_approved":true}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        capstone._load_mapping(artifact)


@pytest.mark.parametrize(
    "raw",
    [
        "production_approved: false\nproduction_approved: true\n",
        (
            "defaults: &defaults\n"
            "  monthly_level_authority: legacy\n"
            "config:\n"
            "  <<: *defaults\n"
            "  monthly_level_authority: solver\n"
        ),
    ],
)
def test_load_mapping_rejects_duplicate_yaml_keys(
    tmp_path: Path,
    raw: str,
) -> None:
    artifact = tmp_path / "ambiguous.yaml"
    artifact.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate YAML key"):
        capstone._load_mapping(artifact)


@pytest.mark.parametrize(
    ("suffix", "literal"),
    [
        (".json", "NaN"),
        (".json", "Infinity"),
        (".json", "-Infinity"),
        (".yaml", ".nan"),
        (".yaml", ".inf"),
        (".yaml", "-.inf"),
    ],
)
def test_load_mapping_rejects_nonfinite_numbers(
    tmp_path: Path,
    suffix: str,
    literal: str,
) -> None:
    artifact = tmp_path / f"nonfinite{suffix}"
    if suffix == ".json":
        artifact.write_text(f'{{"value": {literal}}}', encoding="utf-8")
    else:
        artifact.write_text(f"value: {literal}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        capstone._load_mapping(artifact)


def test_load_mapping_rejects_cyclic_yaml_alias(tmp_path: Path) -> None:
    artifact = tmp_path / "cyclic.yaml"
    artifact.write_text("value: &self [*self]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cyclic structured-data alias"):
        capstone._load_mapping(artifact)


def test_load_mapping_validates_shared_yaml_alias_dag_once(tmp_path: Path) -> None:
    artifact = tmp_path / "alias-dag.yaml"
    rows = ["a0: &a0 [0]"]
    rows.extend(f"a{index}: &a{index} [*a{index - 1}, *a{index - 1}]" for index in range(1, 41))
    rows.append("root: *a40")
    artifact.write_text("\n".join(rows) + "\n", encoding="utf-8")

    payload = capstone._load_mapping(artifact)

    assert payload["root"] is payload["a40"]


def test_load_mapping_counts_cached_alias_subtree_depth(tmp_path: Path) -> None:
    artifact = tmp_path / "alias-depth.yaml"
    rows = ["s0: &s0 [0]"]
    rows.extend(f"s{index}: &s{index} [*s{index - 1}]" for index in range(1, 90))
    rows.append("root: " + "[" * 50 + "*s89" + "]" * 50)
    artifact.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nesting is too deep"):
        capstone._load_mapping(artifact)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_load_mapping_rejects_excessive_nesting(
    tmp_path: Path,
    suffix: str,
) -> None:
    artifact = tmp_path / f"deep{suffix}"
    if suffix == ".json":
        raw = '{"value":' * 140 + "0" + "}" * 140
    else:
        raw = "value: " + "[" * 140 + "0" + "]" * 140 + "\n"
    artifact.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="nesting is too deep|invalid YAML"):
        capstone._load_mapping(artifact)


@pytest.fixture(autouse=True)
def _trusted_promotion_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FMV_DATA_ROOT", raising=False)
    monkeypatch.delenv("PFC_SHARED_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        capstone,
        "_promotion_now_utc",
        lambda: pd.Timestamp("2026-06-17T00:00:00Z"),
    )
    monkeypatch.setattr(
        capstone,
        "verify_external_operation_receipt",
        lambda payload: payload,
    )
    governance_key = Ed25519PrivateKey.generate()
    governance_private = tmp_path / "governance-private.pem"
    governance_public = tmp_path / "governance-public.pem"
    governance_private.write_bytes(
        governance_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    governance_public.write_bytes(
        governance_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH", str(governance_private))
    monkeypatch.setenv("PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH", str(governance_public))
    monkeypatch.setenv(
        "PFC_LT_DATA_ROOT",
        str(tmp_path.parent / f"{tmp_path.name}-shared-data"),
    )
    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path.parent / f"{tmp_path.name}-promotion-private.pem"
    public_path = tmp_path.parent / f"{tmp_path.name}-promotion-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH", str(private_path))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(public_path))
    acquisition_key = Ed25519PrivateKey.generate()
    acquisition_private = tmp_path.parent / f"{tmp_path.name}-acquisition-private.pem"
    acquisition_public = tmp_path.parent / f"{tmp_path.name}-acquisition-public.pem"
    acquisition_private.write_bytes(
        acquisition_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    acquisition_public.write_bytes(
        acquisition_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH", str(acquisition_private))
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        str(acquisition_public),
    )
    timestamp_key = Ed25519PrivateKey.generate()
    timestamp_private = tmp_path.parent / f"{tmp_path.name}-timestamp-private.pem"
    timestamp_public = tmp_path.parent / f"{tmp_path.name}-timestamp-public.pem"
    timestamp_private.write_bytes(
        timestamp_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    timestamp_public.write_bytes(
        timestamp_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH", str(timestamp_private))
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH", str(timestamp_public))
    journal_key = Ed25519PrivateKey.generate()
    journal_private = tmp_path.parent / f"{tmp_path.name}-journal-private.pem"
    journal_public = tmp_path.parent / f"{tmp_path.name}-journal-public.pem"
    journal_private.write_bytes(
        journal_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    journal_public.write_bytes(
        journal_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH", str(journal_private))
    monkeypatch.setenv("PFC_DATA_JOURNAL_TRUSTED_PUBLIC_KEY_PATH", str(journal_public))
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_JOURNAL_ID", "capstone-source-journal")
    for prefix, private_env, public_env in (
        (
            "publication-request",
            PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV,
            PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV,
        ),
        (
            "publication-anchor",
            "PFC_TEST_DATA_PUBLICATION_ANCHOR_PRIVATE_KEY_PATH",
            PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
        ),
    ):
        publication_key = Ed25519PrivateKey.generate()
        publication_private = tmp_path.parent / f"{tmp_path.name}-{prefix}-private.pem"
        publication_public = tmp_path.parent / f"{tmp_path.name}-{prefix}-public.pem"
        publication_private.write_bytes(
            publication_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        publication_public.write_bytes(
            publication_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setenv(private_env, str(publication_private))
        monkeypatch.setenv(public_env, str(publication_public))
    for index, public_env in enumerate(
        (
            "PFC_DATA_PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_BASE_MODEL_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_SELECTION_INPUT_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_TIMESTAMP_JOURNAL_WITNESS_TRUSTED_PUBLIC_KEY_PATH",
        )
    ):
        registry_key = Ed25519PrivateKey.generate()
        registry_public = tmp_path.parent / f"{tmp_path.name}-global-trust-{index}.pem"
        registry_public.write_bytes(
            registry_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setenv(public_env, str(registry_public))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "8f150d1a-7d21-4c36-a050-b06d734be2d1",
    )


def _external_publication_projection(
    data_root: Path,
    *,
    generation_id: str,
    contract_sha256: str,
) -> dict[str, bytes | str]:
    operation_id = "00000000-0000-4000-8000-000000000201"
    intent = sign_snapshot_publication_intent(
        {
            "schema_version": "lt_snapshot_publication_intent.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "operation_id": operation_id,
            "operation_created_at_utc": "2026-06-16T23:59:20+00:00",
            "transition_type": "PUBLISH",
            "expected_anchor_receipt_id": "0" * 64,
            "expected_anchor_sequence": 1,
            "generation_id": generation_id,
            "contract_path": (
                Path("snapshots") / generation_id / "lt_input_snapshot.json"
            ).as_posix(),
            "contract_sha256": contract_sha256,
            "generation_inventory_sha256": "a" * 64,
            "legacy_pointer_sha256": None,
            "bootstrap_authorization": None,
            "bootstrap_authorization_id": None,
            "bootstrap_authorization_sha256": None,
        },
        private_key_path=os.environ[PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV],
    )
    intent_payload = canonical_json(intent)
    receipt = sign_snapshot_anchor_receipt(
        {
            "schema_version": "lt_snapshot_anchor_receipt.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "sequence": 2,
            "previous_receipt_id": "0" * 64,
            "operation_id": operation_id,
            "intent_id": intent["intent_id"],
            "intent_sha256": hashlib.sha256(intent_payload).hexdigest(),
            "transition_type": "PUBLISH",
            "generation_id": generation_id,
            "contract_sha256": contract_sha256,
            "generation_inventory_sha256": "a" * 64,
            "legacy_pointer_sha256": None,
            "bootstrap_authorization_id": None,
            "bootstrap_authorization_sha256": None,
            "committed_at_utc": "2026-06-16T23:59:25+00:00",
        },
        private_key_path=os.environ["PFC_TEST_DATA_PUBLICATION_ANCHOR_PRIVATE_KEY_PATH"],
    )
    receipt_payload = canonical_json(receipt)
    pointer_payload = canonical_json(
        pointer_mapping_from_external_receipt(
            intent,
            receipt,
            receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        )
    )
    pointer = json.loads(pointer_payload)
    challenge_nonce = "4" * 64
    observation = sign_snapshot_anchor_head_observation(
        {
            "schema_version": "lt_snapshot_anchor_head_observation.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "receipt_id": receipt["receipt_id"],
            "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "sequence": receipt["sequence"],
            "challenge_nonce": challenge_nonce,
            "challenge_sha256": external_head_challenge_sha256(
                pointer,
                challenge_nonce=challenge_nonce,
            ),
            "observed_at_utc": "2026-06-16T23:59:30+00:00",
            "expires_at_utc": "2026-06-17T00:04:30+00:00",
        },
        private_key_path=os.environ["PFC_TEST_DATA_PUBLICATION_ANCHOR_PRIVATE_KEY_PATH"],
    )
    observation_payload = canonical_json(observation)
    view = data_root / "views" / "pfc_lt"
    for path, payload in (
        (view / "intents" / f"{intent['intent_id']}.json", intent_payload),
        (view / "receipts" / f"{receipt['receipt_id']}.json", receipt_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (view / "current.json").write_bytes(pointer_payload)
    return {
        "pointer": pointer_payload,
        "intent": intent_payload,
        "receipt": receipt_payload,
        "observation": observation_payload,
        "challenge_nonce": challenge_nonce,
    }


def _sign_v2_input_contract(
    contract: dict[str, object],
) -> dict[str, object]:
    acquisition_id = str(contract["acquisition_id"])
    files = contract["files"]
    assert isinstance(files, dict)
    enriched: dict[str, object] = {}
    previous: str | None = None
    for sequence, (role, raw_entry) in enumerate(sorted(files.items()), start=1):
        assert isinstance(raw_entry, dict)
        entry = dict(raw_entry)
        source_system = str(entry["source_system"])
        raw_artifact = dict(
            entry.get("raw_artifact")
            or {
                "path": f"raw/{role}.bin",
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
        )
        receipt = sign_source_acquisition_receipt(
            {
                "schema_version": "source_acquisition_receipt.v2",
                "journal_id": "capstone-source-journal",
                "sequence": sequence,
                "previous_receipt_id": previous,
                "acquisition_id": acquisition_id,
                "source_role": role,
                "source_system": source_system,
                "source_locator": f"fixture://{role}",
                "acquisition_method": "PROSPECTIVE_DIRECT",
                "received_at_utc": entry["available_at_utc"],
                "raw_sha256": raw_artifact["sha256"],
                "raw_size_bytes": raw_artifact["size_bytes"],
            },
            private_key_path=os.environ["PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH"],
        )
        previous = str(receipt["receipt_id"])
        entry["source_receipt"] = receipt
        entry["raw_artifact"] = raw_artifact
        entry.setdefault(
            "derivation",
            {
                "parser_code_path": f"parser/{role}.py",
                "parser_code_sha256": "a" * 64,
                "parser_config_path": f"parser/{role}.json",
                "parser_config_sha256": "b" * 64,
                "derived_at_utc": entry["available_at_utc"],
            },
        )
        entry.setdefault(
            "quality_evidence",
            {
                "status": "PASS",
                "policy_sha256": "c" * 64,
                "report_path": f"quality/{role}.json",
                "report_sha256": "d" * 64,
            },
        )
        enriched[role] = entry
    ordered_receipts = [
        entry["source_receipt"] for entry in enriched.values() if isinstance(entry, dict)
    ]
    receipt_ids = [str(receipt["receipt_id"]) for receipt in ordered_receipts]
    checkpoint = sign_source_journal_checkpoint(
        {
            "schema_version": "source_journal_checkpoint.v1",
            "journal_id": "capstone-source-journal",
            "acquisition_id": acquisition_id,
            "first_sequence": 1,
            "last_sequence": len(receipt_ids),
            "previous_receipt_id": None,
            "receipt_ids": receipt_ids,
            "head_receipt_id": receipt_ids[-1],
            "issued_at_utc": contract["available_at_utc"],
            "bundle_root_sha256": governed_snapshot_bundle_root_sha256(enriched),
        },
        private_key_path=os.environ["PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH"],
    )
    unsigned = dict(contract)
    unsigned["schema_version"] = "lt_input_snapshot.v2"
    unsigned["source_journal_checkpoint"] = checkpoint
    unsigned["files"] = enriched
    return sign_acquisition_contract(
        unsigned,
        private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
    )


def test_manifest_backed_promotion_passes_with_matching_real_hashes(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 0
    augmented = pd.read_csv(paths["augmented"])
    strict = augmented[
        augmented["gate_id"].isin(
            {
                "lambda_calibration_artifact_present",
                "production_export_path_parity",
                "selected_config_manifest_parity",
                "selected_config_production_approval",
                "delivered_product_normalization_audit",
                "lt_probabilistic_output_governance",
                "ompex_training_selection_exclusion",
            }
        )
    ]
    assert set(strict["status"]) == {"PASS"}


def test_capstone_rejects_cross_role_collision_in_global_authority_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setenv(
        "PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
        os.environ[PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV],
    )

    with pytest.raises(
        ValueError,
        match="candidate external publication evidence is invalid or not authoritative",
    ):
        _run_capstone(paths)


def test_manifest_backed_promotion_fails_when_anchor_lookup_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)

    def _unavailable(_: bytes) -> bytes:
        raise SnapshotAnchorClientError("external anchor request failed")

    monkeypatch.setattr(capstone, "verify_external_operation_receipt", _unavailable)

    with pytest.raises(ValueError, match="not authoritative"):
        main(
            [
                "--audit-gates",
                str(paths["audit_gates"]),
                "--historical-thresholds",
                str(paths["historical_thresholds"]),
                "--production-manifest",
                str(paths["production_manifest"]),
                "--export-manifest",
                str(paths["export_manifest"]),
                "--selected-config-artifact",
                str(paths["selected_config"]),
                "--data-root",
                str(paths["data_root"]),
                "--product-normalization-summary",
                str(paths["product_summary"]),
                "--augmented-audit-gates",
                str(paths["augmented"]),
            ]
        )


def test_manifest_backed_promotion_blocks_uncalibrated_interval_export(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    export = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))
    export["probabilistic_status"] = "ADVISORY_UNCALIBRATED"
    export["interval_columns"] = ["p10", "p90"]
    export["columns"].extend(["p10", "p90"])
    production = json.loads(paths["production_manifest"].read_text(encoding="utf-8"))
    selected = json.loads(paths["selected_config"].read_text(encoding="utf-8"))
    gates = capstone.build_manifest_governance_gates(
        run_timestamp=pd.Timestamp("2026-06-17T00:00:00Z"),
        production_manifest=production,
        export_manifest=export,
        selected_config=selected,
        active_config_hash=_active_config_hash(production),
        selected_config_hash=selected["config_hash"],
    )
    row = gates[gates["gate_id"].eq("lt_probabilistic_output_governance")].iloc[0]
    assert row["status"] == "CRITICAL"


def test_manifest_backed_promotion_blocks_ompex_in_selection_lineage(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    selected = json.loads(paths["selected_config"].read_text(encoding="utf-8"))
    exclusion = selected["ompex_training_selection_exclusion"]
    exclusion["status"] = "FAIL"
    exclusion["matches"] = ["input_source:ompex_hfc"]
    production = json.loads(paths["production_manifest"].read_text(encoding="utf-8"))
    export = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))
    gates = capstone.build_manifest_governance_gates(
        run_timestamp=pd.Timestamp("2026-06-17T00:00:00Z"),
        production_manifest=production,
        export_manifest=export,
        selected_config=selected,
        active_config_hash=_active_config_hash(production),
        selected_config_hash=selected["config_hash"],
    )
    row = gates[gates["gate_id"].eq("ompex_training_selection_exclusion")].iloc[0]
    assert row["status"] == "CRITICAL"


def test_nonregistered_capstone_and_nonassembled_promotion_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_root = tmp_path / "releases"
    bundle = release_root / "candidates" / "test-run"
    paths = _write_inputs(bundle)
    monkeypatch.setenv("PFC_LT_DATA_ROOT", str(paths["data_root"]))
    receipt = tmp_path / "capstone-receipt.json"
    event_key = Ed25519PrivateKey.generate()
    event_private = tmp_path / "event-private.pem"
    event_public = tmp_path / "event-public.pem"
    event_private.write_bytes(
        event_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    event_public.write_bytes(
        event_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", str(event_private))
    monkeypatch.setenv("PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH", str(event_public))
    monkeypatch.setenv("PFC_PROMOTION_JOURNAL_ROOT", str(tmp_path / "journal"))
    monkeypatch.setenv(
        "PFC_PROMOTION_RELEASE_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174002",
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_promotion_now_utc",
        lambda: datetime(2026, 6, 17, 0, 5, tzinfo=timezone.utc),
    )

    request_id = "test-request"
    operation_created_at_utc = "2026-06-17T00:05:00+00:00"
    with pytest.raises(ValueError, match="authorization context is incomplete"):
        _run_capstone(
            paths,
            output=receipt,
            release_request_id=request_id,
            release_operation_id=request_id,
            expected_current_event_id="NONE",
            operation_created_at_utc=operation_created_at_utc,
            release_root_sha256=atomic_promotion._release_root_id(release_root),
        )
    assert not receipt.exists()
    for private_key_env in (
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(private_key_env, raising=False)
    with pytest.raises(PromotionError, match="permanently disabled"):
        promote_candidate(
            release_root,
            run_id="test-run",
            promotion_receipt_path=receipt,
            expected_current_event_id=None,
            operation_id=request_id,
            release_request_id=request_id,
            operation_created_at_utc=operation_created_at_utc,
            require_assembled_evidence=False,
        )


def test_registered_request_must_reside_in_governed_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-001"
    workflow = tmp_path / "workflow"
    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "register-private.pem"
    public_path = tmp_path / "register-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_RELEASE_WORKFLOW_DOMAIN_ID", "123e4567-e89b-42d3-a456-426614174003")
    monkeypatch.setenv("PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH", str(public_path))
    workflow, workflow_domain_sha256 = ensure_workflow_domain(workflow, create=True)
    payload: dict[str, object] = {
        "schema_version": "governed_lt_release_request.v3",
        "run_id": run_id,
        "registration_contract": "assembled_candidate_seal.v1",
        "workflow_domain_sha256": workflow_domain_sha256,
    }
    request = sign_release_request(payload, private_key_path=private_path)
    request_id = str(request["request_id"])
    canonical = workflow / run_id / "requests" / request_id[:32] / "release_request.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(json.dumps(request), encoding="utf-8")
    outside = tmp_path / "outside" / request_id[:32] / "release_request.json"
    outside.parent.mkdir(parents=True)
    outside.write_text(json.dumps(request), encoding="utf-8")

    loaded = capstone._load_canonical_registered_release_request(
        canonical,
        workflow_root=workflow,
        run_id=run_id,
    )
    assert loaded == request
    with pytest.raises(ValueError, match="outside its canonical workflow-domain location"):
        capstone._load_canonical_registered_release_request(
            outside,
            workflow_root=workflow,
            run_id=run_id,
        )


def test_registered_request_requires_exact_schema(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "register-private.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    unsigned: dict[str, object] = {
        "schema_version": "governed_lt_release_request.v1",
        "run_id": "run-001",
    }

    with pytest.raises(RuntimeError, match="only governed_lt_release_request.v3"):
        sign_release_request(unsigned, private_key_path=private_path)


def test_manifest_backed_promotion_blocks_export_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, export_solution_hash="different_solution")

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    parity = augmented[augmented["gate_id"].eq("production_export_path_parity")].iloc[0]
    assert parity["status"] == "CRITICAL"


def test_manifest_backed_promotion_blocks_non_prod_selected_config(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, selected_production_approved=False)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    approval = augmented[augmented["gate_id"].eq("selected_config_production_approval")].iloc[0]
    assert approval["status"] == "CRITICAL"


def test_manifest_backed_promotion_blocks_candidate_scope_selected_config(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, production_promotion_approved=False)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    approval = augmented[augmented["gate_id"].eq("selected_config_production_approval")].iloc[0]
    assert approval["status"] == "CRITICAL"
    assert "production_promotion_approved=False" in approval["evidence"]


def test_manifest_backed_promotion_blocks_missing_candidate_scope_selected_config(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, include_production_promotion_approved=False)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    approval = augmented[augmented["gate_id"].eq("selected_config_production_approval")].iloc[0]
    assert approval["status"] == "CRITICAL"
    assert "production_promotion_approved=None" in approval["evidence"]


def test_manifest_backed_promotion_blocks_negative_selection_status(tmp_path: Path) -> None:
    for selection_status in (
        "NOT_PRODUCTION_APPROVED",
        "DIAGNOSTIC_SELECTED_NOT_PRODUCTION_APPROVED",
        "production_approved",
        "Production_Approved",
    ):
        paths = _write_inputs(
            tmp_path / selection_status.lower(),
            selected_production_approved=True,
            selection_status=selection_status,
        )

        rc = main(
            [
                "--audit-gates",
                str(paths["audit_gates"]),
                "--historical-thresholds",
                str(paths["historical_thresholds"]),
                "--production-manifest",
                str(paths["production_manifest"]),
                "--export-manifest",
                str(paths["export_manifest"]),
                "--selected-config-artifact",
                str(paths["selected_config"]),
                "--data-root",
                str(paths["data_root"]),
                "--product-normalization-summary",
                str(paths["product_summary"]),
                "--augmented-audit-gates",
                str(paths["augmented"]),
            ]
        )

        assert rc == 1
        augmented = pd.read_csv(paths["augmented"])
        approval = augmented[augmented["gate_id"].eq("selected_config_production_approval")].iloc[0]
        assert approval["status"] == "CRITICAL"


def test_manifest_backed_promotion_blocks_selected_solution_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, selected_solution_hash="stale_solution")

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    parity = augmented[augmented["gate_id"].eq("selected_config_manifest_parity")].iloc[0]
    assert parity["status"] == "CRITICAL"
    assert "monthly_solution_hash" in parity["evidence"]


def test_manifest_backed_promotion_blocks_selected_constraints_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, selected_constraints_hash="stale_constraints")

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    parity = augmented[augmented["gate_id"].eq("selected_config_manifest_parity")].iloc[0]
    assert parity["status"] == "CRITICAL"
    assert "active_constraints_hash" in parity["evidence"]


def test_manifest_backed_promotion_blocks_missing_delivered_product_summary(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    delivered = augmented[augmented["gate_id"].eq("delivered_product_normalization_audit")].iloc[0]
    assert delivered["status"] == "CRITICAL"
    assert "missing delivered product normalization summary" in delivered["evidence"]


def test_manifest_backed_promotion_blocks_failed_delivered_product_summary(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path, product_summary_overrides={"all_gates_pass": False, "critical_count": 1}
    )

    rc = main(
        [
            "--audit-gates",
            str(paths["audit_gates"]),
            "--historical-thresholds",
            str(paths["historical_thresholds"]),
            "--production-manifest",
            str(paths["production_manifest"]),
            "--export-manifest",
            str(paths["export_manifest"]),
            "--selected-config-artifact",
            str(paths["selected_config"]),
            "--data-root",
            str(paths["data_root"]),
            "--product-normalization-summary",
            str(paths["product_summary"]),
            "--augmented-audit-gates",
            str(paths["augmented"]),
        ]
    )

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    delivered = augmented[augmented["gate_id"].eq("delivered_product_normalization_audit")].iloc[0]
    assert delivered["status"] == "CRITICAL"
    assert "critical_count_nonzero" in delivered["evidence"]


def test_manifest_backed_promotion_blocks_product_summary_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths["product_summary"].read_text(encoding="utf-8"))
    payload["status_counts"] = {"PASS": 13}
    paths["product_summary"].write_text(json.dumps(payload), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    rc = main(_main_args(paths))

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    delivered = augmented[augmented["gate_id"].eq("delivered_product_normalization_audit")].iloc[0]
    assert delivered["status"] == "CRITICAL"
    assert "summary_sha256_mismatch" in delivered["evidence"]


def test_capstone_replays_product_audit_after_coherent_summary_rehash(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    summary = json.loads(paths["product_summary"].read_text(encoding="utf-8"))
    summary["supported_hard_gate_max_abs_residual_eur_mwh"] = 999.0
    paths["product_summary"].write_text(json.dumps(summary), encoding="utf-8")
    summary_hash = hashlib.sha256(paths["product_summary"].read_bytes()).hexdigest()

    for role in ("production_manifest", "export_manifest"):
        manifest = json.loads(paths[role].read_text(encoding="utf-8"))
        manifest["delivered_product_normalization_evidence"]["summary_sha256"] = summary_hash
        paths[role].write_text(json.dumps(manifest), encoding="utf-8")
    selected = json.loads(paths["selected_config"].read_text(encoding="utf-8"))
    selected["delivered_product_normalization_evidence"]["summary_sha256"] = summary_hash
    selected["candidate_manifest_sha256"] = hashlib.sha256(
        paths["production_manifest"].read_bytes()
    ).hexdigest()
    paths["selected_config"].write_text(json.dumps(selected), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    assert _run_capstone(paths) == 1
    delivered = (
        pd.read_csv(paths["augmented"])
        .query("gate_id == 'delivered_product_normalization_audit'")
        .iloc[0]
    )
    assert delivered["status"] == "CRITICAL"
    assert "product_replay_summary_mismatch" in delivered["evidence"]


def test_manifest_backed_promotion_blocks_missing_export_product_evidence(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    payload = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))
    payload.pop("delivered_product_normalization_evidence")
    paths["export_manifest"].write_text(json.dumps(payload), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    rc = main(_main_args(paths))

    assert rc == 1
    augmented = pd.read_csv(paths["augmented"])
    delivered = augmented[augmented["gate_id"].eq("delivered_product_normalization_audit")].iloc[0]
    assert delivered["status"] == "CRITICAL"
    assert "export_delivered_product_normalization_evidence_missing" in delivered["evidence"]


def _main_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--audit-gates",
        str(paths["audit_gates"]),
        "--historical-thresholds",
        str(paths["historical_thresholds"]),
        "--production-manifest",
        str(paths["production_manifest"]),
        "--export-manifest",
        str(paths["export_manifest"]),
        "--selected-config-artifact",
        str(paths["selected_config"]),
        "--data-root",
        str(paths["data_root"]),
        "--product-normalization-summary",
        str(paths["product_summary"]),
        "--augmented-audit-gates",
        str(paths["augmented"]),
    ]


def test_manifest_backed_promotion_recomputes_canonical_config_hash(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    selected = json.loads(paths["selected_config"].read_text(encoding="utf-8"))
    selected["canonical_config"]["lambda_shape"] = 999.0
    paths["selected_config"].write_text(json.dumps(selected), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    rc = _run_capstone(paths)

    assert rc == 1
    parity = (
        pd.read_csv(paths["augmented"])
        .query("gate_id == 'selected_config_manifest_parity'")
        .iloc[0]
    )
    assert "selected_config_hash" in parity["evidence"]


def test_manifest_backed_promotion_rehashes_candidate_manifest(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    selected = json.loads(paths["selected_config"].read_text(encoding="utf-8"))
    selected["candidate_manifest_sha256"] = "0" * 64
    paths["selected_config"].write_text(json.dumps(selected), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    rc = _run_capstone(paths)

    assert rc == 1
    parity = (
        pd.read_csv(paths["augmented"])
        .query("gate_id == 'selected_config_manifest_parity'")
        .iloc[0]
    )
    assert "candidate_manifest_sha256" in parity["evidence"]


def test_capstone_rejects_partial_promotion_identity(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    receipt_path = tmp_path.parent / f"{tmp_path.name}-promotion_receipt.json"

    with pytest.raises(ValueError, match="authorization context is incomplete"):
        _run_capstone(
            paths,
            output=receipt_path,
            release_request_id="request-001",
            evidence_inventory_sha256="a" * 64,
        )

    assert not receipt_path.exists()


def test_manifest_backed_promotion_blocks_input_mutation_during_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    original = capstone.evaluate_monthly_curve_promotion

    def mutate_after_evaluation(*args, **kwargs):
        decision = original(*args, **kwargs)
        paths["production_manifest"].write_text("{}", encoding="utf-8")
        return decision

    monkeypatch.setattr(capstone, "evaluate_monthly_curve_promotion", mutate_after_evaluation)

    with pytest.raises(ValueError, match="changed during evaluation"):
        _run_capstone(
            paths,
            output=tmp_path.parent / f"{tmp_path.name}-receipt.json",
        )

    assert not (tmp_path.parent / f"{tmp_path.name}-receipt.json").exists()


@pytest.mark.parametrize(
    "artifact_role",
    ["delivered_csv", "delivered_forwards", "monthly_candidate"],
)
def test_product_replay_uses_captured_bytes_during_swap_restore_attack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_role: str,
) -> None:
    paths = _write_inputs(tmp_path)
    target = paths[artifact_role]
    original_bytes = target.read_bytes()
    original_run_audit = capstone.run_audit

    def swap_restore(*args, **kwargs):
        target.write_bytes(b"attacker-controlled-transient-bytes")
        try:
            return original_run_audit(*args, **kwargs)
        finally:
            target.write_bytes(original_bytes)

    monkeypatch.setattr(capstone, "run_audit", swap_restore)

    assert _run_capstone(paths) == 0


def test_capstone_rejects_coherently_rehashed_permissive_hierarchy_policy(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    for role in ("production_manifest", "export_manifest"):
        manifest = json.loads(paths[role].read_text(encoding="utf-8"))
        policy = dict(manifest["product_hierarchy_policy"])
        policy.update(
            {
                "priority": ["CALENDAR", "QUARTER", "MONTH"],
                "quote_conflict_tolerance_eur_mwh": 999.0,
                "hard_repricing_tolerance_eur_mwh": 999.0,
                "residual_lineage_required": False,
            }
        )
        manifest["product_hierarchy_policy"] = policy
        manifest["product_hierarchy_policy_sha256"] = hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        paths[role].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    assert _run_capstone(paths) == 1


def test_capstone_rejects_invented_constraint_provenance_with_rehashed_payloads(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    for role in ("production_manifest", "export_manifest"):
        manifest = json.loads(paths[role].read_text(encoding="utf-8"))
        rows = list(manifest["constraint_provenance_rows"])
        row = dict(rows[0])
        payload = json.loads(row["lineage_payload"])
        payload["parent_quote_id"] = "f" * 64
        payload["source_quote_ids"] = ["f" * 64]
        row["source_quote_ids"] = "f" * 64
        row["lineage_payload"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        row["lineage_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rows[0] = row
        manifest["constraint_provenance_rows"] = rows
        manifest["constraint_provenance_hash"] = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        paths[role].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    assert _run_capstone(paths) == 1


def test_capstone_reconstructs_hard_quotes_from_verified_forward_snapshot(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    for role in ("production_manifest", "export_manifest"):
        manifest = json.loads(paths[role].read_text(encoding="utf-8"))
        manifest["hard_quotes"][0]["price_eur_mwh"] += 5.0
        manifest["hard_quote_set_hash"] = hashlib.sha256(
            json.dumps(manifest["hard_quotes"], sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        row = manifest["constraint_provenance_rows"][0]
        payload = json.loads(row["lineage_payload"])
        payload["target"] += 5.0
        row["target"] += 5.0
        row["lineage_payload"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        row["lineage_sha256"] = hashlib.sha256(row["lineage_payload"].encode("utf-8")).hexdigest()
        manifest["constraint_provenance_hash"] = hashlib.sha256(
            json.dumps(
                manifest["constraint_provenance_rows"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        paths[role].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    assert _run_capstone(paths) == 1
    gate = (
        pd.read_csv(paths["augmented"])
        .query("gate_id == 'delivered_product_normalization_audit'")
        .iloc[0]
    )
    assert "hard_quotes_forward_snapshot_mismatch" in gate["evidence"]


def test_constraint_governance_recomputes_residual_energy_identity() -> None:
    month_quote = {
        "product": "2027-01",
        "load_type": "BASE",
        "price_eur_mwh": 90.0,
        "quote_id": "month-quote-id",
    }
    year_quote = {
        "product": "2027",
        "load_type": "BASE",
        "price_eur_mwh": 81.0,
        "quote_id": "year-quote-id",
    }
    hard_quotes = [month_quote, year_quote]
    direct_payload = {
        "formula": "direct_parent_mean",
        "parent_product": "2027-01",
        "parent_quote_id": "month-quote-id",
        "source_quote_ids": ["month-quote-id"],
        "known_buckets": [],
        "target": 90.0,
    }
    residual_target = (81.0 * 8760.0 - 90.0 * 744.0) / 8016.0
    residual_payload = {
        "formula": "(parent_price*parent_hours-known_bucket_energy)/free_hours",
        "parent_product": "2027",
        "parent_quote_id": "year-quote-id",
        "source_quote_ids": ["year-quote-id", "month-quote-id"],
        "known_buckets": ["2027-01"],
        "target": residual_target,
    }
    rows = [
        {
            "constraint_name": "BASE:2027-01",
            "product": "2027-01",
            "parent_product": "2027-01",
            "source_quote_ids": "month-quote-id",
            "lineage_formula": "direct_parent_mean",
            "lineage_sha256": capstone._sha256_json(direct_payload),
            "lineage_payload": json.dumps(direct_payload, sort_keys=True, separators=(",", ":")),
            "target": 90.0,
            "hours": 744.0,
            "n_months": 1,
            "is_residual": False,
            "load_type": "BASE",
        },
        {
            "constraint_name": "BASE:2027-RESIDUAL",
            "product": "2027-RESIDUAL",
            "parent_product": "2027",
            "source_quote_ids": "year-quote-id|month-quote-id",
            "lineage_formula": "residual_parent_energy_less_known_buckets",
            "lineage_sha256": capstone._sha256_json(residual_payload),
            "lineage_payload": json.dumps(residual_payload, sort_keys=True, separators=(",", ":")),
            "target": residual_target,
            "hours": 8016.0,
            "n_months": 11,
            "is_residual": True,
            "load_type": "BASE",
        },
    ]
    policy = {
        "schema_version": "monthly_quote_hierarchy_policy.v1",
        "priority": ["MONTH", "QUARTER", "CALENDAR"],
        "quote_conflict_tolerance_eur_mwh": 0.01,
        "hard_repricing_tolerance_eur_mwh": 1e-9,
        "stationarity_tolerance": 1e-7,
        "residual_lineage_required": True,
    }
    manifest = {
        "delivery_months": [
            str(month) for month in pd.period_range("2027-01", "2027-12", freq="M")
        ],
        "solver_config": {
            "quote_conflict_tolerance": 0.01,
            "constraint_tolerance": 1e-9,
            "stationarity_tolerance": 1e-7,
        },
        "solver_kkt": {
            "max_abs_constraint_residual": 1e-12,
            "stationarity_residual": 1e-12,
            "condition_number": 100.0,
            "ridge_used": False,
            "solved_by_lstsq": False,
        },
        "product_hierarchy_policy": policy,
        "product_hierarchy_policy_sha256": capstone._sha256_json(policy),
        "forward_snapshot": {"quotes": hard_quotes},
        "hard_quotes": hard_quotes,
        "hard_quote_set_hash": capstone._sha256_json(hard_quotes),
        "constraint_provenance_rows": rows,
        "constraint_provenance_hash": capstone._sha256_json(rows),
    }
    _, errors = capstone._verify_constraint_governance(manifest)
    assert errors == []

    rows[0]["hours"] = 1.0
    rows[1]["hours"] = 1.0
    manifest["constraint_provenance_hash"] = capstone._sha256_json(rows)
    _, errors = capstone._verify_constraint_governance(manifest)
    assert any("calendar_hours_mismatch" in error for error in errors)
    rows[0]["hours"] = 744.0
    rows[1]["hours"] = 8016.0

    rows[1]["target"] = residual_target + 1.0
    residual_payload["target"] = residual_target + 1.0
    rows[1]["lineage_payload"] = json.dumps(residual_payload, sort_keys=True, separators=(",", ":"))
    rows[1]["lineage_sha256"] = capstone._sha256_json(residual_payload)
    manifest["constraint_provenance_hash"] = capstone._sha256_json(rows)
    _, errors = capstone._verify_constraint_governance(manifest)
    assert any("residual_energy_identity_mismatch" in error for error in errors)

    duplicated_target = (81.0 * 8760.0 - 2.0 * 90.0 * 744.0) / 7272.0
    rows[1]["target"] = duplicated_target
    rows[1]["hours"] = 7272.0
    residual_payload["target"] = duplicated_target
    residual_payload["known_buckets"] = ["2027-01", "2027-01"]
    rows[1]["lineage_payload"] = json.dumps(residual_payload, sort_keys=True, separators=(",", ":"))
    rows[1]["lineage_sha256"] = capstone._sha256_json(residual_payload)
    manifest["constraint_provenance_hash"] = capstone._sha256_json(rows)
    _, errors = capstone._verify_constraint_governance(manifest)
    assert any("known_buckets_duplicate" in error for error in errors)
    assert any("known_bucket_overlap" in error for error in errors)


def test_constraint_governance_requires_every_hard_quote_parent() -> None:
    quotes = [
        {
            "product": "2027-01",
            "load_type": "BASE",
            "price_eur_mwh": 90.0,
            "quote_id": "month-quote-id",
        },
        {
            "product": "2027",
            "load_type": "BASE",
            "price_eur_mwh": 81.0,
            "quote_id": "year-quote-id",
        },
    ]
    manifest = _manifest(
        monthly_solution_hash="solution",
        forward_binding=({"quotes": quotes}, {}),
    )
    manifest["constraint_provenance_rows"] = manifest["constraint_provenance_rows"][:1]
    manifest["constraint_provenance_hash"] = capstone._sha256_json(
        manifest["constraint_provenance_rows"]
    )

    _, errors = capstone._verify_constraint_governance(manifest)

    assert any("hard_quote_partition_incomplete" in error for error in errors)


def test_constraint_governance_rejects_duplicate_product_rows() -> None:
    quote = {
        "product": "2027-01",
        "load_type": "BASE",
        "price_eur_mwh": 90.0,
        "quote_id": "month-quote-id",
    }
    manifest = _manifest(
        monthly_solution_hash="solution",
        forward_binding=({"quotes": [quote]}, {}),
    )
    manifest["constraint_provenance_rows"].append(dict(manifest["constraint_provenance_rows"][0]))
    manifest["constraint_provenance_hash"] = capstone._sha256_json(
        manifest["constraint_provenance_rows"]
    )

    _, errors = capstone._verify_constraint_governance(manifest)

    assert "constraint_provenance_product_duplicate" in errors
    assert any("constraint_provenance_global_overlap" in error for error in errors)


def test_constraint_governance_accepts_redundant_consistent_parent_quote() -> None:
    months = pd.period_range("2027-01", "2027-12", freq="M")
    quotes = [
        MarketQuote(
            market="CH",
            product=str(month),
            load_type="BASE",
            price=80.0,
            quote_id=f"month-{month}",
        )
        for month in months
    ]
    quotes.append(
        MarketQuote(
            market="CH",
            product="2027",
            load_type="BASE",
            price=80.0,
            quote_id="cal-2027",
        )
    )
    system = build_monthly_constraint_system(months, quotes)
    hard_quotes = [
        {
            "product": quote.product,
            "load_type": quote.load_type,
            "price_eur_mwh": quote.price,
            "quote_id": quote.quote_id,
        }
        for quote in quotes
    ]
    provenance_columns = [
        "constraint_name",
        "product",
        "parent_product",
        "source_quote_ids",
        "lineage_formula",
        "lineage_sha256",
        "lineage_payload",
        "target",
        "hours",
        "n_months",
        "is_residual",
        "load_type",
    ]
    rows = system.rows[provenance_columns].to_dict(orient="records")
    policy = {
        "schema_version": "monthly_quote_hierarchy_policy.v1",
        "priority": ["MONTH", "QUARTER", "CALENDAR"],
        "quote_conflict_tolerance_eur_mwh": 0.01,
        "hard_repricing_tolerance_eur_mwh": 1e-9,
        "stationarity_tolerance": 1e-7,
        "residual_lineage_required": True,
    }
    manifest = {
        "delivery_months": [str(month) for month in months],
        "solver_config": {
            "quote_conflict_tolerance": 0.01,
            "constraint_tolerance": 1e-9,
            "stationarity_tolerance": 1e-7,
        },
        "solver_kkt": {
            "max_abs_constraint_residual": 1e-12,
            "stationarity_residual": 1e-12,
            "condition_number": 100.0,
            "ridge_used": False,
            "solved_by_lstsq": False,
        },
        "product_hierarchy_policy": policy,
        "product_hierarchy_policy_sha256": capstone._sha256_json(policy),
        "forward_snapshot": {"quotes": hard_quotes},
        "hard_quotes": hard_quotes,
        "hard_quote_set_hash": capstone._sha256_json(hard_quotes),
        "constraint_provenance_rows": rows,
        "constraint_provenance_hash": capstone._sha256_json(rows),
    }

    _, errors = capstone._verify_constraint_governance(manifest)

    assert errors == []


def test_capstone_rejects_different_prod_export_constraint_provenance(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    export = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))
    export["constraint_provenance_rows"][0]["hours"] = 9999.0
    export["constraint_provenance_hash"] = hashlib.sha256(
        json.dumps(
            export["constraint_provenance_rows"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    paths["export_manifest"].write_text(json.dumps(export), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    assert _run_capstone(paths) == 1


def test_capstone_rejects_tampered_file_inside_candidate_bundle(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    paths["delivered_csv"].write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate bundle verification failed"):
        _run_capstone(paths)


def test_capstone_rejects_forward_eligibility_replayed_at_later_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(
        capstone,
        "_promotion_now_utc",
        lambda: pd.Timestamp("2026-07-13T00:00:00Z"),
    )

    assert _run_capstone(paths) == 1


def test_capstone_rejects_removed_run_timestamp_cli_override(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "--audit-gates",
                str(paths["audit_gates"]),
                "--historical-thresholds",
                str(paths["historical_thresholds"]),
                "--production-manifest",
                str(paths["production_manifest"]),
                "--export-manifest",
                str(paths["export_manifest"]),
                "--selected-config-artifact",
                str(paths["selected_config"]),
                "--data-root",
                str(paths["data_root"]),
                "--run-timestamp",
                "2026-06-17T00:00:00Z",
            ]
        )


def test_capstone_requires_hash_bound_candidate_run_manifest(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    paths["candidate_run_manifest"].unlink()
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    with pytest.raises(ValueError, match="candidate run manifest is missing"):
        _run_capstone(paths)


def test_capstone_rejects_noneligible_data_contract(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    def mutate(contract: dict[str, object]) -> None:
        contract["calibration_eligible"] = False

    _replace_candidate_data_contract(paths, mutate=mutate, resign=True)

    with pytest.raises(ValueError, match="data_contract_not_calibration_eligible"):
        _run_capstone(paths)


def test_capstone_rejects_bootstrap_publication_explicitly(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    bundle = paths["candidate_run_manifest"].parent.parent
    intent_path = bundle / "evidence" / "data" / "snapshot_publication_intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["transition_type"] = "BOOTSTRAP"
    intent_path.write_text(json.dumps(intent), encoding="utf-8")
    run_manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    run_manifest["data_publication_intent"]["sha256"] = hashlib.sha256(
        intent_path.read_bytes()
    ).hexdigest()
    paths["candidate_run_manifest"].write_text(
        json.dumps(run_manifest),
        encoding="utf-8",
    )
    _write_candidate_bundle_manifest(
        paths["candidate_bundle_manifest"].parent,
        paths["candidate_bundle_manifest"],
    )

    with pytest.raises(
        ValueError,
        match="capstone cannot consume a BOOTSTRAP publication",
    ):
        _run_capstone(paths)


def test_capstone_rejects_self_declared_governed_acquisition(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    def mutate(contract: dict[str, object]) -> None:
        contract["provenance"] = "FORGED_GOVERNED_ACQUISITION"

    _replace_candidate_data_contract(paths, mutate=mutate, resign=False)

    with pytest.raises(ValueError, match="external publication evidence is invalid"):
        _run_capstone(paths)


def test_capstone_rejects_role_available_after_valuation(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    def mutate(contract: dict[str, object]) -> None:
        files = contract["files"]
        assert isinstance(files, dict)
        role = files["epex_ch"]
        assert isinstance(role, dict)
        role["available_at_utc"] = "2026-06-18T00:00:00+00:00"

    _replace_candidate_data_contract(paths, mutate=mutate, resign=True)

    with pytest.raises(ValueError, match="input_source_epex_ch_available_after_valuation"):
        _run_capstone(paths)


def test_capstone_rejects_role_cutoff_divergent_from_contract(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    def mutate(contract: dict[str, object]) -> None:
        files = contract["files"]
        assert isinstance(files, dict)
        role = files["epex_ch"]
        assert isinstance(role, dict)
        role["available_at_utc"] = "2026-06-16T23:59:59+00:00"

    _replace_candidate_data_contract(paths, mutate=mutate, resign=True)

    with pytest.raises(ValueError, match="data_contract_acquisition_attestation"):
        _run_capstone(paths)


def test_capstone_derives_enabled_outages_role_from_archived_config(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    config_path = tmp_path / "evidence" / "config" / "config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quality"]["freshness"]["outages_enabled"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    paths["candidate_run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    with pytest.raises(ValueError, match="input_sources_missing_consumed_roles.*outages"):
        _run_capstone(paths)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("outages", "quality.freshness.outages_enabled must be a boolean"),
        ("solver", "forwards.monthly_curve_solver.enabled must be a boolean"),
    ],
)
def test_capstone_rejects_truthy_non_boolean_role_flags(
    tmp_path: Path,
    section: str,
    expected: str,
) -> None:
    paths = _write_inputs(tmp_path)
    config_path = tmp_path / "evidence" / "config" / "config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if section == "outages":
        config["quality"]["freshness"]["outages_enabled"] = 1
    else:
        config["forwards"]["monthly_curve_solver"]["enabled"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    paths["candidate_run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    with pytest.raises(ValueError, match=expected):
        _run_capstone(paths)


def test_candidate_sealing_rejects_serialization_before_valuation(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    manifest["candidate_serialized_at_utc"] = "2026-06-16T23:59:59+00:00"
    paths["candidate_run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CandidateEvidenceError, match="timestamps are inconsistent"):
        _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])


def test_candidate_cadence_rejects_sparse_two_hour_series() -> None:
    index = pd.date_range("2026-06-10T00:00:00Z", periods=80, freq="2h")
    frame = pd.DataFrame({"price_eur_mwh": 80.0}, index=index)

    errors = capstone._candidate_cadence_errors(frame, role="epex_ch")

    assert any(error.startswith("epex_ch.grid_coverage=") for error in errors)


def test_candidate_cadence_rejects_dense_but_phase_shifted_series() -> None:
    index = pd.date_range("2026-06-10T00:07:00Z", periods=96, freq="15min")
    frame = pd.DataFrame({"price_eur_mwh": 80.0}, index=index)

    errors = capstone._candidate_cadence_errors(frame, role="epex_ch")

    assert "epex_ch.grid_phase_mismatch" in errors


def test_capstone_replays_optional_input_freshness_at_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    reference = pd.Timestamp("2026-06-17T00:00:00Z")
    index = pd.date_range(reference - pd.Timedelta(days=7), reference, freq="15min")
    frame = pd.DataFrame(
        {"unavailable_mw": 100.0, "n_outages": 1.0},
        index=index,
    )
    _add_optional_input_role(paths, role="outages", frame=frame, reference=reference)
    monkeypatch.setattr(
        capstone,
        "_promotion_now_utc",
        lambda: pd.Timestamp("2026-06-19T01:00:00Z"),
    )

    assert _run_capstone(paths) == 1
    gate = pd.read_csv(paths["augmented"]).query("gate_id == 'candidate_run_freshness'").iloc[0]
    assert "outages.promotion_recompute_failed" in gate["evidence"]


def test_capstone_rejects_optional_role_cutoff_divergence(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    reference = pd.Timestamp("2026-06-17T00:00:00Z")
    index = pd.date_range(reference - pd.Timedelta(days=7), reference, freq="15min")
    frame = pd.DataFrame(
        {"unavailable_mw": 100.0, "n_outages": 1.0},
        index=index,
    )
    _add_optional_input_role(paths, role="outages", frame=frame, reference=reference)

    def mutate(contract: dict[str, object]) -> None:
        files = contract["files"]
        assert isinstance(files, dict)
        role = files["outages"]
        assert isinstance(role, dict)
        role["available_at_utc"] = "2026-06-16T23:59:59Z"

    _replace_candidate_data_contract(paths, mutate=mutate, resign=True)

    with pytest.raises(ValueError, match="data_contract_acquisition_attestation"):
        _run_capstone(paths)


def test_capstone_rejects_semantically_future_governed_eex_history(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    reference = pd.Timestamp("2026-06-17T00:00:00Z")
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-18"]),
            "product": ["2027"],
            "load_type": ["BASE"],
            "product_type": ["Cal"],
            "price": [80.0],
            "market": ["CH"],
            "source": ["EEX"],
        }
    )
    _add_governed_eex_history_role(paths, frame=history, reference=reference)

    with pytest.raises(
        ValueError,
        match="data_contract_available_after_valuation.*"
        "input_source_eex_forwards_history_available_after_valuation",
    ):
        _run_capstone(paths)


def test_capstone_rejects_solver_config_without_solver_manifest_authority(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    reference = pd.Timestamp("2026-06-17T00:00:00Z")
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-16"]),
            "product": ["2027"],
            "load_type": ["BASE"],
            "product_type": ["Cal"],
            "price": [80.0],
            "market": ["CH"],
            "source": ["EEX"],
        }
    )
    _add_governed_eex_history_role(paths, frame=history, reference=reference)
    production = json.loads(paths["production_manifest"].read_text(encoding="utf-8"))
    production.pop("monthly_level_authority", None)
    paths["production_manifest"].write_text(json.dumps(production), encoding="utf-8")
    _write_candidate_bundle_manifest(
        paths["candidate_bundle_manifest"].parent,
        paths["candidate_bundle_manifest"],
    )

    with pytest.raises(ValueError, match="must be solver or legacy"):
        _run_capstone(paths)


def test_capstone_replays_candidate_input_freshness_at_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(
        capstone,
        "_promotion_now_utc",
        lambda: pd.Timestamp("2026-06-17T02:00:00Z"),
    )

    assert _run_capstone(paths) == 1
    gate = pd.read_csv(paths["augmented"]).query("gate_id == 'candidate_run_freshness'").iloc[0]
    assert gate["status"] == "CRITICAL"
    assert "gap_at_promotion" in gate["evidence"]


def test_capstone_rejects_relative_data_root(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        _run_capstone(paths, data_root="relative/shared-data")


def test_capstone_rejects_empty_data_root_environment(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        _run_capstone(paths, data_root="")


def test_capstone_refuses_outputs_inside_finalized_bundle(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="outside finalized candidate bundle"):
        _run_capstone(paths, output=tmp_path / "receipt.json")


def test_capstone_rejects_coherently_rehashed_forged_freshness_report(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    manifest["freshness_reports"]["epex_ch"]["checks_passed"] = 999
    paths["candidate_run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    assert _run_capstone(paths) == 1
    gate = pd.read_csv(paths["augmented"]).query("gate_id == 'candidate_run_freshness'").iloc[0]
    assert "declared_freshness_report_mismatch" in gate["evidence"]


def test_capstone_rejects_legacy_external_pointer_identity(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    manifest["data_root_id"] = "legacy_external_pointer"
    manifest["data_pointer"]["logical_path"] = "current.json"
    paths["candidate_run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(tmp_path, paths["candidate_bundle_manifest"])

    with pytest.raises(
        ValueError,
        match="data_pointer_not_canonical_lt_view,data_root_id",
    ):
        _run_capstone(paths)


def test_capstone_rejects_live_generation_mutated_after_candidate(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    source = (
        Path(str(paths["data_root"]))
        / "snapshots"
        / "governed-generation-001"
        / "inputs"
        / "epex_ch.parquet"
    )
    frame = pd.read_parquet(source)
    frame.iloc[-1, 0] = 999.0
    frame.to_parquet(source)

    assert _run_capstone(paths) == 1
    gate = pd.read_csv(paths["augmented"]).query("gate_id == 'candidate_run_freshness'").iloc[0]
    assert any(
        message in gate["evidence"]
        for message in (
            "supporting artifact hash mismatch",
            "snapshot artifact epex_ch exceeds the maximum size",
        )
    )


def test_monthly_solver_history_binding_requires_same_governed_bytes() -> None:
    source_hash = "a" * 64
    catalog_hash = "b" * 64
    history, source_summary = _governed_history_evidence()
    capstone._verify_governed_monthly_solver_history_binding(
        production_manifest={
            "monthly_level_authority": "solver",
            "source_hashes": {
                "eex_forwards_history": source_hash,
                "eex_forwards_history_source_summary": capstone._sha256_json(source_summary),
                "eex_historical_vintage_catalog": catalog_hash,
                "eex_forwards_history_consumed_frame": dataframe_sha256(history),
            },
            "eex_forwards_history_source_summary": source_summary,
            "verified_vintage_catalog": history.attrs["verified_vintage_catalog"],
        },
        candidate_run_manifest={
            "input_sources": {
                "eex_forwards_history": {"sha256": source_hash},
            }
        },
        data_contract={
            "files": {
                "eex_forwards_history": {
                    "sha256": source_hash,
                    "calibration_eligible": True,
                    "vintage_catalog": {"sha256": catalog_hash},
                }
            }
        },
        active_config={
            "forwards": {"monthly_curve_solver": {"enabled": True, "target_markets": ["CH"]}}
        },
        governed_history=history,
        valuation_timestamp=pd.Timestamp("2026-06-17T00:00:00Z"),
    )


def test_monthly_solver_history_binding_rejects_consumed_frame_mutation() -> None:
    source_hash = "a" * 64
    catalog_hash = "b" * 64
    history, source_summary = _governed_history_evidence()
    with pytest.raises(ValueError, match="consumed EEX history frame hash mismatch"):
        capstone._verify_governed_monthly_solver_history_binding(
            production_manifest={
                "monthly_level_authority": "solver",
                "source_hashes": {
                    "eex_forwards_history": source_hash,
                    "eex_forwards_history_source_summary": capstone._sha256_json(source_summary),
                    "eex_historical_vintage_catalog": catalog_hash,
                    "eex_forwards_history_consumed_frame": "d" * 64,
                },
                "eex_forwards_history_source_summary": source_summary,
                "verified_vintage_catalog": history.attrs["verified_vintage_catalog"],
            },
            candidate_run_manifest={
                "input_sources": {"eex_forwards_history": {"sha256": source_hash}}
            },
            data_contract={
                "files": {
                    "eex_forwards_history": {
                        "sha256": source_hash,
                        "calibration_eligible": True,
                        "vintage_catalog": {"sha256": catalog_hash},
                    }
                }
            },
            active_config={
                "forwards": {"monthly_curve_solver": {"enabled": True, "target_markets": ["CH"]}}
            },
            governed_history=history,
            valuation_timestamp=pd.Timestamp("2026-06-17T00:00:00Z"),
        )


@pytest.mark.parametrize("missing_role", ["summary", "summary_hash"])
def test_monthly_solver_history_binding_requires_recomputed_source_summary(
    missing_role: str,
) -> None:
    source_hash = "a" * 64
    catalog_hash = "b" * 64
    history, source_summary = _governed_history_evidence()
    production: dict[str, object] = {
        "monthly_level_authority": "solver",
        "source_hashes": {
            "eex_forwards_history": source_hash,
            "eex_forwards_history_source_summary": capstone._sha256_json(source_summary),
            "eex_historical_vintage_catalog": catalog_hash,
            "eex_forwards_history_consumed_frame": dataframe_sha256(history),
        },
        "eex_forwards_history_source_summary": source_summary,
        "verified_vintage_catalog": history.attrs["verified_vintage_catalog"],
    }
    if missing_role == "summary":
        production.pop("eex_forwards_history_source_summary")
    else:
        production["source_hashes"].pop("eex_forwards_history_source_summary")
    with pytest.raises(ValueError, match="source summary"):
        capstone._verify_governed_monthly_solver_history_binding(
            production_manifest=production,
            candidate_run_manifest={
                "input_sources": {"eex_forwards_history": {"sha256": source_hash}}
            },
            data_contract={
                "files": {
                    "eex_forwards_history": {
                        "sha256": source_hash,
                        "calibration_eligible": True,
                        "vintage_catalog": {"sha256": catalog_hash},
                    }
                }
            },
            active_config={
                "forwards": {
                    "monthly_curve_solver": {
                        "enabled": True,
                        "target_markets": ["CH"],
                    }
                }
            },
            governed_history=history,
            valuation_timestamp=pd.Timestamp("2026-06-17T00:00:00Z"),
        )


@pytest.mark.parametrize(
    ("receipt_hash", "contract_hash", "eligible", "message"),
    [
        ("b" * 64, "a" * 64, True, "hash binding mismatch"),
        ("a" * 64, "b" * 64, True, "hash binding mismatch"),
        ("a" * 64, "a" * 64, False, "not calibration eligible"),
    ],
)
def test_monthly_solver_history_binding_rejects_substitution(
    receipt_hash: str,
    contract_hash: str,
    eligible: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        capstone._verify_governed_monthly_solver_history_binding(
            production_manifest={
                "monthly_level_authority": "solver",
                "source_hashes": {"eex_forwards_history": "a" * 64},
            },
            candidate_run_manifest={
                "input_sources": {
                    "eex_forwards_history": {"sha256": receipt_hash},
                }
            },
            data_contract={
                "files": {
                    "eex_forwards_history": {
                        "sha256": contract_hash,
                        "calibration_eligible": eligible,
                    }
                }
            },
            active_config={
                "forwards": {
                    "monthly_curve_solver": {
                        "enabled": True,
                        "target_markets": ["CH"],
                    }
                }
            },
        )


@pytest.mark.parametrize(
    ("solver_enabled", "manifest_authority"),
    [(True, "legacy"), (False, "solver")],
)
def test_monthly_solver_history_binding_rejects_config_authority_divergence(
    solver_enabled: bool,
    manifest_authority: str,
) -> None:
    source_hash = "a" * 64
    with pytest.raises(ValueError, match="solver activation does not match"):
        capstone._verify_governed_monthly_solver_history_binding(
            production_manifest={
                "monthly_level_authority": manifest_authority,
                "source_hashes": {"eex_forwards_history": source_hash},
            },
            candidate_run_manifest={
                "input_sources": {
                    "eex_forwards_history": {"sha256": source_hash},
                }
            },
            data_contract={
                "files": {
                    "eex_forwards_history": {
                        "sha256": source_hash,
                        "calibration_eligible": True,
                    }
                }
            },
            active_config={
                "forwards": {
                    "monthly_curve_solver": {
                        "enabled": solver_enabled,
                        "target_markets": ["CH"],
                    }
                }
            },
        )


def test_monthly_solver_history_binding_rejects_unknown_authority() -> None:
    with pytest.raises(ValueError, match="must be solver or legacy"):
        capstone._verify_governed_monthly_solver_history_binding(
            production_manifest={"monthly_level_authority": "GARBAGE"},
            candidate_run_manifest={"input_sources": {}},
            data_contract={"files": {}},
            active_config={
                "forwards": {
                    "monthly_curve_solver": {
                        "enabled": False,
                        "target_markets": ["CH"],
                    }
                }
            },
        )


def test_capstone_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    receipt = tmp_path.parent / f"{tmp_path.name}-immutable-receipt.json"

    assert _run_capstone(paths, output=receipt) == 0
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_capstone(paths, output=receipt)
    assert receipt.read_bytes() == before


def _run_capstone(
    paths: dict[str, Path],
    *,
    data_root: Path | str | None = None,
    output: Path | None = None,
    release_request_id: str | None = None,
    release_operation_id: str | None = None,
    expected_current_event_id: str | None = None,
    operation_created_at_utc: str | None = None,
    release_root_sha256: str | None = None,
    evidence_inventory_sha256: str | None = None,
) -> int:
    args = [
        "--audit-gates",
        str(paths["audit_gates"]),
        "--historical-thresholds",
        str(paths["historical_thresholds"]),
        "--production-manifest",
        str(paths["production_manifest"]),
        "--export-manifest",
        str(paths["export_manifest"]),
        "--selected-config-artifact",
        str(paths["selected_config"]),
        "--data-root",
        str(paths["data_root"] if data_root is None else data_root),
        "--product-normalization-summary",
        str(paths["product_summary"]),
        "--augmented-audit-gates",
        str(paths["augmented"]),
    ]
    if output is not None:
        args.extend(["--output", str(output)])
    if release_request_id is not None:
        args.extend(["--release-request-id", release_request_id])
    if release_operation_id is not None:
        args.extend(["--release-operation-id", release_operation_id])
    if expected_current_event_id is not None:
        args.extend(["--expected-current-event-id", expected_current_event_id])
    if operation_created_at_utc is not None:
        args.extend(["--operation-created-at-utc", operation_created_at_utc])
    if release_root_sha256 is not None:
        args.extend(["--release-root-sha256", release_root_sha256])
    if evidence_inventory_sha256 is not None:
        args.extend(["--evidence-inventory-sha256", evidence_inventory_sha256])
    return main(args)


def _write_inputs(
    tmp_path: Path,
    *,
    export_solution_hash: str = "solution",
    selected_production_approved: bool = True,
    production_promotion_approved: bool | None = None,
    selection_status: str | None = None,
    selected_solution_hash: str = "solution",
    selected_constraints_hash: str = "constraints",
    include_production_promotion_approved: bool = True,
    product_summary_overrides: dict[str, object] | None = None,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    audit_gates = tmp_path / "audit_gates.csv"
    historical_thresholds = tmp_path / "historical_thresholds.csv"
    production_manifest = tmp_path / "production_manifest.json"
    export_manifest = tmp_path / "export_manifest.json"
    selected_config = tmp_path / "candidate_config.json"
    product_summary = tmp_path / "product_normalization_summary.json"
    monthly_candidate = tmp_path / "monthly_candidate_manifest.json"
    candidate_run_manifest = tmp_path / "manifests" / "candidate_run_manifest.json"
    candidate_bundle_manifest = tmp_path / "candidate_bundle_manifest.json"
    delivered_csv = tmp_path / "pfc_15min.csv"
    delivered_forwards = tmp_path / "forwards.parquet"
    augmented = tmp_path.parent / f"{tmp_path.name}-augmented_audit_gates.csv"

    _audit_gates().to_csv(audit_gates, index=False)
    _historical_threshold_fixture().to_csv(historical_thresholds, index=False)
    forward_binding = _forward_binding_fixture(tmp_path)
    candidate_payload = _manifest(
        monthly_solution_hash="solution",
        forward_binding=forward_binding,
    )
    candidate_payload["active_config_hash"] = _active_config_hash(candidate_payload)
    candidate_payload["promotion_eligible"] = True
    monthly_candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")
    _write_delivered_product_fixture(
        delivered_csv,
        delivered_forwards,
        candidate_payload["forward_snapshot"],
    )
    _, product_payload = run_audit(
        csv_path=delivered_csv,
        forwards_path=delivered_forwards,
        market="CH",
        required_forward_date="2026-06-17",
        required_load_types=("BASE", "PEAK"),
        monthly_candidate_manifest_path=monthly_candidate,
        command_argv=["audit_ch_product_normalization.py", "TEST_REPLAY_FIXTURE"],
    )
    if product_summary_overrides:
        product_payload.update(product_summary_overrides)
    product_summary.write_text(json.dumps(product_payload), encoding="utf-8")
    product_evidence = {
        "summary_sha256": hashlib.sha256(product_summary.read_bytes()).hexdigest(),
        "input_csv_sha256": product_payload["input_csv_sha256"],
        "forwards_sha256": product_payload["forwards_sha256"],
        "forward_snapshot_date": product_payload["forward_snapshot_date"],
        "market": product_payload["market"],
    }
    production = _manifest(monthly_solution_hash="solution", forward_binding=forward_binding)
    export = _manifest(monthly_solution_hash=export_solution_hash, forward_binding=forward_binding)
    export.update(
        {
            "delivered_curve_kind": "deterministic_price_shape",
            "probabilistic_status": "DETERMINISTIC_ONLY",
            "interval_columns": [],
            "intervals_permitted_for_pricing_or_risk": False,
            "source_price_column": "price_shape",
            "scenario_weights": {"price_shape": 1.0},
            "producer_contract": "pfc.pipeline.candidate_hourly_export.v1",
            "columns": [
                "timestamp_ch",
                "utc_offset_ch",
                "timestamp_utc",
                "price_weighted_mean_eur_mwh",
            ],
            "finite_price_values": True,
        }
    )
    production["delivered_product_normalization_evidence"] = dict(product_evidence)
    export["delivered_product_normalization_evidence"] = dict(product_evidence)
    production_manifest.write_text(json.dumps(production), encoding="utf-8")
    export_manifest.write_text(json.dumps(export), encoding="utf-8")
    selected_payload = {
        "schema_version": "monthly_curve_selected_config.v1",
        "config_hash": _active_config_hash(production),
        "canonical_config": _active_config_payload(production),
        "active_config_hash_from_candidate_manifest": _active_config_hash(production),
        "monthly_solution_hash": selected_solution_hash,
        "active_constraints_hash": selected_constraints_hash,
        "candidate_manifest": str(production_manifest),
        "candidate_manifest_sha256": hashlib.sha256(production_manifest.read_bytes()).hexdigest(),
        "production_approved": selected_production_approved,
        "delivered_product_normalization_evidence": dict(product_evidence),
        "selection_status": (
            selection_status
            if selection_status is not None
            else "PRODUCTION_APPROVED"
            if selected_production_approved
            else "DIAGNOSTIC_SELECTED_NOT_PRODUCTION_APPROVED"
        ),
    }
    if include_production_promotion_approved:
        selected_payload["production_promotion_approved"] = (
            True if production_promotion_approved is None else production_promotion_approved
        )
    selected_payload["ompex_training_selection_exclusion"] = {
        "schema_version": "ompex_training_selection_exclusion.v1",
        "status": "PASS",
        "prohibited_uses": [
            "MODEL_INPUT",
            "TARGET",
            "LOSS",
            "SELECTION",
            "PROMOTION_GATE",
        ],
        "inspected_input_sources": [{"role": "epex_ch", "identity": "epex_ch"}],
        "inspected_selected_config_sha256": selected_payload["config_hash"],
        "matches": [],
    }
    selected_config.write_text(json.dumps(selected_payload), encoding="utf-8")
    _write_candidate_run_manifest(tmp_path, candidate_run_manifest)
    _write_candidate_bundle_manifest(tmp_path, candidate_bundle_manifest)
    return {
        "audit_gates": audit_gates,
        "historical_thresholds": historical_thresholds,
        "production_manifest": production_manifest,
        "export_manifest": export_manifest,
        "selected_config": selected_config,
        "product_summary": product_summary,
        "monthly_candidate": monthly_candidate,
        "candidate_run_manifest": candidate_run_manifest,
        "candidate_bundle_manifest": candidate_bundle_manifest,
        "delivered_csv": delivered_csv,
        "delivered_forwards": delivered_forwards,
        "augmented": augmented,
        "data_root": tmp_path.parent / f"{tmp_path.name}-shared-data",
    }


def _historical_threshold_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gate_id, metric in (
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        (
            "residual_vs_implied_comparable_block",
            "comparable_block_shape_delta_abs_eur_mwh",
        ),
    ):
        for bucket in ["all", *(f"month_{month:02d}" for month in range(1, 13))]:
            rows.append(
                {
                    "gate_id": gate_id,
                    "metric": metric,
                    "market": "CH",
                    "delivery_bucket": bucket,
                    "lookback_start": "",
                    "lookback_end": "",
                    "n_snapshots": 0,
                    "min_required_n": 24,
                    "p50": None,
                    "p90": None,
                    "p975": None,
                    "max_observed": None,
                    "regime_filter": "test_fixture",
                    "status": "UNSUPPORTED",
                }
            )
    return pd.DataFrame(rows)


def _write_candidate_run_manifest(bundle: Path, manifest_path: Path) -> None:
    generation_id = "governed-generation-001"
    acquisition_id = "governed-acquisition-001"
    reference = pd.Timestamp("2026-06-17T00:00:00Z")
    role_columns = {
        "epex_ch": ("price_eur_mwh",),
        "epex_de": ("price_eur_mwh",),
        "entso": ("load_mw", "solar_mw", "wind_mw"),
        "hydro": ("fill_deviation", "water_value_supported"),
    }
    data_root = bundle.parent / f"{bundle.name}-shared-data"
    generation_root = data_root / "snapshots" / generation_id
    generation_root.mkdir(parents=True, exist_ok=True)
    contract_files: dict[str, object] = {}
    input_sources: dict[str, object] = {}
    freshness_reports: dict[str, object] = {}
    for role, columns in role_columns.items():
        if role == "hydro":
            index = pd.date_range(
                end=reference.tz_convert("Europe/Zurich").normalize(),
                periods=312,
                freq="W-MON",
            ).tz_convert("UTC")
            fill_pct = [
                42.0 + float(position % 52) * 0.2 + float(position // 52) * 0.5
                for position in range(len(index))
            ]
            bronze = pd.DataFrame(
                {
                    "fill_pct": fill_pct,
                    "fill_gwh": [value * 100.0 for value in fill_pct],
                    "max_capacity_gwh": 10000.0,
                },
                index=index,
            )
            frame = ingest_hydro.build_water_value(bronze)
        else:
            index = pd.date_range(reference - pd.Timedelta(days=7), reference, freq="15min")
            bronze = pd.DataFrame(
                {
                    column: [float(position + 1)] * len(index)
                    for position, column in enumerate(columns)
                },
                index=index,
            )
            frame = (
                ingest_entso.build_features(bronze)
                if role == "entso"
                else ingest_epex._clean(bronze)
            )
        logical_path = f"inputs/{role}.parquet"
        source_path = generation_root / logical_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(source_path)
        payload = source_path.read_bytes()
        source_hash = hashlib.sha256(payload).hexdigest()
        raw_path = generation_root / "raw" / f"{role}.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        bronze.to_parquet(raw_path)
        raw_payload = raw_path.read_bytes()
        parser_code_path = generation_root / "parser" / f"{role}.py"
        parser_code_path.parent.mkdir(parents=True, exist_ok=True)
        parser_code_path.write_bytes(approved_parser_path_for_role(role).read_bytes())
        source_system = {
            "epex_ch": "EPEX_SPOT",
            "epex_de": "EPEX_SPOT",
            "entso": "ENTSOE_TRANSPARENCY_PLATFORM",
            "hydro": "FMV_HYDRO_CURATED",
        }[role]
        parser_config_path = generation_root / "parser" / f"{role}.json"
        parser_config_path.write_text(
            json.dumps(
                build_replay_config(
                    role=role,
                    source_system=source_system,
                    raw_frame=bronze,
                    derived_frame=frame,
                )
            ),
            encoding="utf-8",
        )
        quality_policy_hash = "c" * 64
        quality_path = generation_root / "quality" / f"{role}.json"
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(
            json.dumps(
                {
                    "schema_version": "lt_source_quality.v1",
                    "role": role,
                    "status": "PASS",
                    "policy_sha256": quality_policy_hash,
                }
            ),
            encoding="utf-8",
        )
        contract_files[role] = {
            "path": logical_path,
            "size_bytes": len(payload),
            "sha256": source_hash,
            "acquisition_id": acquisition_id,
            "available_at_utc": reference.isoformat(),
            "calibration_eligible": True,
            "expected_cadence_seconds": 604800 if role == "hydro" else 900,
            "source_system": source_system,
            "raw_artifact": {
                "path": raw_path.relative_to(generation_root).as_posix(),
                "size_bytes": len(raw_payload),
                "sha256": hashlib.sha256(raw_payload).hexdigest(),
            },
            "derivation": {
                "parser_code_path": parser_code_path.relative_to(generation_root).as_posix(),
                "parser_code_sha256": hashlib.sha256(parser_code_path.read_bytes()).hexdigest(),
                "parser_config_path": parser_config_path.relative_to(generation_root).as_posix(),
                "parser_config_sha256": hashlib.sha256(parser_config_path.read_bytes()).hexdigest(),
                "derived_at_utc": reference.isoformat(),
            },
            "quality_evidence": {
                "status": "PASS",
                "policy_sha256": quality_policy_hash,
                "report_path": quality_path.relative_to(generation_root).as_posix(),
                "report_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
            },
        }
        input_sources[role] = {
            "role": role,
            "logical_path": logical_path,
            "size_bytes": len(payload),
            "sha256": source_hash,
            "rows": len(frame),
            "frame_sha256": dataframe_sha256(frame),
        }
        report = validate_input_frame(
            frame,
            name=role,
            required_columns=list(columns),
            min_rows=1,
            max_age_days=10.0 if role == "hydro" else 3.0 if role == "entso" else 2.0,
            reference_timestamp=reference,
            fail_on_stale=True,
            min_finite_fraction=0.95,
            recent_window_days=7.0,
            min_recent_finite_fraction=0.95,
            max_finite_gap_days=14.0
            if role == "hydro"
            else 2.0 / 24.0
            if role == "entso"
            else 1.0 / 24.0,
        )
        freshness_reports[role] = {
            "dataset": report.dataset,
            "checks_passed": report.checks_passed,
            "warnings": report.warnings,
            "errors": report.errors,
            "metrics": report.metrics,
        }

    live_contract_path = generation_root / "lt_input_snapshot.json"
    contract = {
        "schema_version": "lt_input_snapshot.v1",
        "layout": "external_v2",
        "generation_id": generation_id,
        "acquisition_id": acquisition_id,
        "created_at_utc": reference.isoformat(),
        "available_at_utc": reference.isoformat(),
        "calibration_eligible": True,
        "source_class": "GOVERNED_ACQUISITION",
        "provenance": "TEST_GOVERNED_ACQUISITION",
        "files": contract_files,
    }
    contract = _sign_v2_input_contract(contract)
    live_contract_path.write_text(json.dumps(contract), encoding="utf-8")
    contract_hash = hashlib.sha256(live_contract_path.read_bytes()).hexdigest()
    data_dir = bundle / "evidence" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    contract_path = data_dir / "lt_input_snapshot.json"
    contract_path.write_bytes(live_contract_path.read_bytes())
    publication = _external_publication_projection(
        data_root,
        generation_id=generation_id,
        contract_sha256=contract_hash,
    )
    pointer_path = data_dir / "current.json"
    pointer_path.write_bytes(publication["pointer"])
    publication_intent = data_dir / "snapshot_publication_intent.json"
    publication_receipt = data_dir / "snapshot_anchor_receipt.json"
    publication_observation = data_dir / "snapshot_anchor_head_observation.json"
    publication_intent.write_bytes(publication["intent"])
    publication_receipt.write_bytes(publication["receipt"])
    publication_observation.write_bytes(publication["observation"])
    config_path = bundle / "evidence" / "config" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "forwards": {
                    "monthly_curve_solver": {
                        "enabled": False,
                        "target_markets": ["CH"],
                    }
                },
                "quality": {"freshness": {"outages_enabled": False}},
            }
        ),
        encoding="utf-8",
    )
    market_artifact = bundle / "markets" / "CH" / "pfc.csv"
    model_artifact = bundle / "models" / "shared" / "shape.bin"
    market_artifact.parent.mkdir(parents=True, exist_ok=True)
    model_artifact.parent.mkdir(parents=True, exist_ok=True)
    market_artifact.write_bytes(b"deterministic-candidate")
    model_artifact.write_bytes(b"deterministic-model")
    deterministic_artifacts = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (market_artifact, model_artifact)
    }
    deterministic_sha256 = hashlib.sha256(
        json.dumps(
            deterministic_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "lt_candidate_run.v1",
                "run_id": "test-run",
                "pipeline_scope": "LT_ONLY",
                "promotion_eligible": False,
                "candidate_serialized_at_utc": reference.isoformat(),
                "run_started_at_utc": reference.isoformat(),
                "reference_timestamp": reference.isoformat(),
                "reference_timestamp_is_explicit": True,
                "data_layout": "external_v2",
                "data_root_id": "fmv_data_view:pfc_lt",
                "data_generation_id": generation_id,
                "config_path": config_path.relative_to(bundle).as_posix(),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "data_contract": {
                    "archive_path": contract_path.relative_to(bundle).as_posix(),
                    "sha256": contract_hash,
                },
                "data_pointer": {
                    "archive_path": pointer_path.relative_to(bundle).as_posix(),
                    "sha256": hashlib.sha256(pointer_path.read_bytes()).hexdigest(),
                    "logical_path": "views/pfc_lt/current.json",
                },
                "data_publication_intent": {
                    "archive_path": publication_intent.relative_to(bundle).as_posix(),
                    "sha256": hashlib.sha256(publication_intent.read_bytes()).hexdigest(),
                },
                "data_publication_anchor_receipt": {
                    "archive_path": publication_receipt.relative_to(bundle).as_posix(),
                    "sha256": hashlib.sha256(publication_receipt.read_bytes()).hexdigest(),
                },
                "data_publication_head_observation": {
                    "archive_path": publication_observation.relative_to(bundle).as_posix(),
                    "sha256": hashlib.sha256(publication_observation.read_bytes()).hexdigest(),
                },
                "input_sources": input_sources,
                "freshness_reports": freshness_reports,
                "deterministic_artifacts": deterministic_artifacts,
                "deterministic_artifacts_sha256": deterministic_sha256,
            }
        ),
        encoding="utf-8",
    )


def _write_candidate_bundle_manifest(bundle: Path, manifest_path: Path) -> None:
    evidence_manifest = bundle / EVIDENCE_MANIFEST
    evidence_manifest.unlink(missing_ok=True)
    run_manifest_path = bundle / "manifests" / "candidate_run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_manifest.pop("candidate_evidence_manifest", None)
        run_manifest.pop("candidate_evidence_manifest_sha256", None)
        run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
        authority = bundle / "evidence" / "promotion_authority"
        authority.mkdir(parents=True, exist_ok=True)
        selected_decision = authority / "selected_lambda_decision.json"
        candidate_config = yaml.safe_load(
            (bundle / "evidence" / "config" / "config.yaml").read_text(encoding="utf-8")
        )
        canonical = active_monthly_curve_config_payload(monthly_solver_settings(candidate_config))
        selected_decision.write_text(
            json.dumps(
                {
                    "schema_version": "monthly_curve_selected_lambda_decision.v1",
                    "decision_id": "decision-test",
                    "calibration_campaign_id": "campaign-test",
                    "selection_reason": "test fixture",
                    "selection_status": "PRODUCTION_APPROVED",
                    "production_approved": True,
                    "canonical_config": canonical,
                    "config_hash": config_hash(canonical),
                }
            ),
            encoding="utf-8",
        )
        threshold_receipt = authority / "historical_thresholds.authority.json"
        selected_receipt = authority / "selected_lambda_decision.authority.json"
        threshold_receipt.write_text(
            json.dumps(
                sign_model_governance_artifact_receipt(
                    bundle / "historical_thresholds.csv",
                    artifact_role="historical_thresholds",
                    authority_id="FMV_MODEL_GOVERNANCE_TEST",
                    issued_at_utc="2026-06-16T00:00:00Z",
                    data_cutoff_utc="2026-06-15T00:00:00Z",
                    private_key_path=os.environ["PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"],
                )
            ),
            encoding="utf-8",
        )
        selected_receipt.write_text(
            json.dumps(
                sign_model_governance_artifact_receipt(
                    selected_decision,
                    artifact_role="selected_lambda_decision",
                    authority_id="FMV_MODEL_GOVERNANCE_TEST",
                    issued_at_utc="2026-06-16T00:00:00Z",
                    data_cutoff_utc="2026-06-15T00:00:00Z",
                    private_key_path=os.environ["PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"],
                )
            ),
            encoding="utf-8",
        )
        _bind_pre_run_evidence_fixture(
            bundle,
            run_manifest_path=run_manifest_path,
            historical_thresholds=bundle / "historical_thresholds.csv",
            historical_thresholds_receipt=threshold_receipt,
            selected_lambda_decision=selected_decision,
            selected_lambda_decision_receipt=selected_receipt,
        )
        seal_candidate_evidence(
            bundle,
            run_id="test-run",
            artifacts={
                "historical_thresholds": EvidenceArtifactInput(
                    path=bundle / "historical_thresholds.csv",
                    producer_contract="test.thresholds.v1",
                    authority_receipt_path=threshold_receipt,
                ),
                "selected_lambda_decision": EvidenceArtifactInput(
                    path=selected_decision,
                    producer_contract="test.selected_lambda.v1",
                    authority_receipt_path=selected_receipt,
                ),
                "production_manifest": EvidenceArtifactInput(
                    path=bundle / "production_manifest.json",
                    producer_contract="test.production_manifest.v1",
                ),
                "export_manifest": EvidenceArtifactInput(
                    path=bundle / "export_manifest.json",
                    producer_contract="test.export_manifest.v1",
                ),
                "audit_gates": EvidenceArtifactInput(
                    path=bundle / "audit_gates.csv",
                    producer_contract="test.audit_gates.v1",
                ),
                "product_normalization_summary": EvidenceArtifactInput(
                    path=bundle / "product_normalization_summary.json",
                    producer_contract="test.product_normalization.v1",
                ),
                "selected_config_run_parity": EvidenceArtifactInput(
                    path=bundle / "candidate_config.json",
                    producer_contract="test.selected_config_parity.v1",
                ),
            },
        )
    files = {
        path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    metadata: dict[str, object] = {}
    content_payload = {"run_id": "test-run", "files": files, "metadata": metadata}
    content_hash = hashlib.sha256(
        json.dumps(content_payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "candidate_bundle.v1",
                "run_id": "test-run",
                "created_at_utc": "2026-06-17T00:00:00+00:00",
                "files": files,
                "metadata": metadata,
                "content_sha256": content_hash,
            }
        ),
        encoding="utf-8",
    )


def _bind_pre_run_evidence_fixture(
    bundle: Path,
    *,
    run_manifest_path: Path,
    historical_thresholds: Path,
    historical_thresholds_receipt: Path,
    selected_lambda_decision: Path,
    selected_lambda_decision_receipt: Path,
) -> None:
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    entries: dict[str, object] = {}
    for role, artifact, receipt in (
        (
            "historical_thresholds",
            historical_thresholds,
            historical_thresholds_receipt,
        ),
        (
            "selected_lambda_decision",
            selected_lambda_decision,
            selected_lambda_decision_receipt,
        ),
    ):
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        entries[role] = {
            "path": artifact.relative_to(bundle).as_posix(),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size_bytes": artifact.stat().st_size,
            "authority_receipt": {
                "path": receipt.relative_to(bundle).as_posix(),
                "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "receipt_id": receipt_payload["receipt_id"],
                "authority_id": receipt_payload["authority_id"],
                "issued_at_utc": receipt_payload["issued_at_utc"],
                "data_cutoff_utc": receipt_payload["data_cutoff_utc"],
            },
        }
    config_path = bundle / str(run["config_path"])
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entries["runtime_config"] = {
        "path": config_path.relative_to(bundle).as_posix(),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "size_bytes": config_path.stat().st_size,
        "semantic_sha256": hashlib.sha256(
            json.dumps(
                config_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    generation_id = str(run["data_generation_id"])
    publication_observation_entry = run["data_publication_head_observation"]
    publication_observation = json.loads(
        (bundle / publication_observation_entry["archive_path"]).read_text(encoding="utf-8")
    )
    run_spec = {
        "schema_version": GOVERNED_RUN_SPEC_SCHEMA,
        "run_id": run["run_id"],
        "as_of_utc": run["reference_timestamp"],
        "build_timestamp_utc": run["candidate_serialized_at_utc"],
        "run_started_at_utc": run["run_started_at_utc"],
        "runtime": {
            "distribution": "fmv-pfc-lt",
            "version": __version__,
            "source_revision": "1" * 40,
        },
        "config": {
            key: entries["runtime_config"][key] for key in ("path", "sha256", "semantic_sha256")
        },
        "input_snapshot": {
            "generation_id": generation_id,
            "contract_logical_path": f"snapshots/{generation_id}/lt_input_snapshot.json",
            "contract_sha256": run["data_contract"]["sha256"],
            "pointer_logical_path": "views/pfc_lt/current.json",
            "pointer_sha256": run["data_pointer"]["sha256"],
            "publication_head_observation_sha256": (publication_observation_entry["sha256"]),
            "publication_head_challenge_nonce": publication_observation["challenge_nonce"],
        },
        "model_controls": {
            "peak_source_policy": "same_first",
            "use_seasonal_hourly_shape": True,
        },
        "pre_run_artifacts": _run_spec_artifact_bindings(entries),
    }
    pre_path = bundle / "manifests" / "pre_run_governance_manifest.json"
    pre_path.write_text(
        json.dumps(
            {
                "schema_version": "pre_run_model_governance.v2",
                "run_id": run["run_id"],
                "valuation_timestamp": run["reference_timestamp"],
                "captured_at_utc": run["candidate_serialized_at_utc"],
                "artifacts": entries,
                "governed_run_spec": run_spec,
                "governed_run_spec_sha256": _canonical_mapping_sha256(run_spec),
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    run["pre_run_governance_manifest"] = "manifests/pre_run_governance_manifest.json"
    run["pre_run_governance_manifest_sha256"] = hashlib.sha256(pre_path.read_bytes()).hexdigest()
    run_manifest_path.write_text(json.dumps(run), encoding="utf-8")


def _add_optional_input_role(
    paths: dict[str, Path],
    *,
    role: str,
    frame: pd.DataFrame,
    reference: pd.Timestamp,
) -> None:
    generation_root = paths["data_root"] / "snapshots" / "governed-generation-001"
    logical_path = f"inputs/{role}.parquet"
    source_path = generation_root / logical_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(source_path)
    payload = source_path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    parser_code_path = generation_root / "parser" / f"{role}.py"
    parser_code_path.parent.mkdir(parents=True, exist_ok=True)
    parser_code_path.write_text("def parse(payload):\n    return payload\n", encoding="utf-8")
    parser_config_path = generation_root / "parser" / f"{role}.json"
    parser_config_path.write_text(
        json.dumps({"role": role, "schema_version": "parser_config.v1"}),
        encoding="utf-8",
    )
    quality_policy_hash = "f" * 64
    quality_path = generation_root / "quality" / f"{role}.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(
        json.dumps(
            {
                "schema_version": "lt_source_quality.v1",
                "role": role,
                "status": "PASS",
                "policy_sha256": quality_policy_hash,
            }
        ),
        encoding="utf-8",
    )

    def mutate(contract: dict[str, object]) -> None:
        files = contract["files"]
        assert isinstance(files, dict)
        existing_receipts = sorted(
            (entry["source_receipt"] for entry in files.values()),
            key=lambda receipt: int(receipt["sequence"]),
        )
        last_receipt = existing_receipts[-1]
        source_receipt = sign_source_acquisition_receipt(
            {
                "schema_version": "source_acquisition_receipt.v2",
                "journal_id": os.environ["PFC_DATA_TIMESTAMP_JOURNAL_ID"],
                "sequence": int(last_receipt["sequence"]) + 1,
                "previous_receipt_id": last_receipt["receipt_id"],
                "acquisition_id": contract["acquisition_id"],
                "source_role": role,
                "source_system": "ENTSOE_TRANSPARENCY_PLATFORM",
                "source_locator": f"fixture://{role}",
                "acquisition_method": "PROSPECTIVE_DIRECT",
                "received_at_utc": reference.isoformat(),
                "raw_sha256": source_hash,
                "raw_size_bytes": len(payload),
            },
            private_key_path=os.environ["PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH"],
        )
        files[role] = {
            "path": logical_path,
            "size_bytes": len(payload),
            "sha256": source_hash,
            "acquisition_id": contract["acquisition_id"],
            "available_at_utc": reference.isoformat(),
            "calibration_eligible": True,
            "expected_cadence_seconds": 900,
            "source_system": "ENTSOE_TRANSPARENCY_PLATFORM",
            "source_receipt": source_receipt,
            "raw_artifact": {
                "path": logical_path,
                "size_bytes": len(payload),
                "sha256": source_hash,
            },
            "derivation": {
                "parser_code_path": parser_code_path.relative_to(generation_root).as_posix(),
                "parser_code_sha256": hashlib.sha256(parser_code_path.read_bytes()).hexdigest(),
                "parser_config_path": parser_config_path.relative_to(generation_root).as_posix(),
                "parser_config_sha256": hashlib.sha256(parser_config_path.read_bytes()).hexdigest(),
                "derived_at_utc": reference.isoformat(),
            },
            "quality_evidence": {
                "status": "PASS",
                "policy_sha256": quality_policy_hash,
                "report_path": quality_path.relative_to(generation_root).as_posix(),
                "report_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
            },
        }
        receipt_ids = [
            str(entry["source_receipt"]["receipt_id"])
            for entry in sorted(
                files.values(),
                key=lambda entry: int(entry["source_receipt"]["sequence"]),
            )
        ]
        contract["source_journal_checkpoint"] = sign_source_journal_checkpoint(
            {
                "schema_version": "source_journal_checkpoint.v1",
                "journal_id": os.environ["PFC_DATA_TIMESTAMP_JOURNAL_ID"],
                "acquisition_id": contract["acquisition_id"],
                "first_sequence": 1,
                "last_sequence": len(receipt_ids),
                "previous_receipt_id": None,
                "receipt_ids": receipt_ids,
                "head_receipt_id": receipt_ids[-1],
                "issued_at_utc": reference.isoformat(),
                "bundle_root_sha256": governed_snapshot_bundle_root_sha256(files),
            },
            private_key_path=os.environ["PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH"],
        )

    _replace_candidate_data_contract(paths, mutate=mutate, resign=True)
    manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    manifest["input_sources"][role] = {
        "role": role,
        "logical_path": logical_path,
        "size_bytes": len(payload),
        "sha256": source_hash,
        "rows": len(frame),
        "frame_sha256": dataframe_sha256(frame),
    }
    report = validate_input_frame(
        frame,
        name=role,
        required_columns=["unavailable_mw", "n_outages"],
        min_rows=1,
        max_age_days=2.0,
        reference_timestamp=reference,
        fail_on_stale=True,
        min_finite_fraction=0.95,
        recent_window_days=7.0,
        min_recent_finite_fraction=0.95,
        max_finite_gap_days=2.0 / 24.0,
    )
    manifest["freshness_reports"][role] = {
        "dataset": report.dataset,
        "checks_passed": report.checks_passed,
        "warnings": report.warnings,
        "errors": report.errors,
        "metrics": report.metrics,
    }
    if role == "outages":
        config_path = (
            paths["candidate_run_manifest"].parent.parent / "evidence" / "config" / "config.yaml"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["quality"]["freshness"]["outages_enabled"] = True
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    paths["candidate_run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    _write_candidate_bundle_manifest(
        paths["candidate_bundle_manifest"].parent,
        paths["candidate_bundle_manifest"],
    )


def _add_governed_eex_history_role(
    paths: dict[str, Path],
    *,
    frame: pd.DataFrame,
    reference: pd.Timestamp,
) -> None:
    role = "eex_forwards_history"
    generation_root = paths["data_root"] / "snapshots" / "governed-generation-001"
    catalog_root = generation_root / "eex-vintages"
    logical_path = "eex-vintages/history.parquet"
    source_path = generation_root / logical_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_date = pd.Timestamp(frame.iloc[0]["date"]).normalize()
    observed = source_date.tz_localize("UTC") + pd.Timedelta(hours=8)
    product = str(frame.iloc[0]["product"]).strip()
    price = float(frame.iloc[0]["price"])
    workbook_path = catalog_root / "sources" / f"eex-{source_date.date()}.xlsx"
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = pd.DataFrame(
        [
            [None, f"Y01_{product}_BASE"],
            [None, "ISIN_fixture"],
            ["Date", None],
            [source_date.strftime("%d.%m.%Y"), price],
        ]
    )
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        workbook.to_excel(writer, sheet_name="CH", index=False, header=False)
    workbook_payload = workbook_path.read_bytes()
    workbook_hash = hashlib.sha256(workbook_payload).hexdigest()
    parser_code_path = catalog_root / "parser" / "ingest_forwards.py"
    parser_code_path.parent.mkdir(parents=True, exist_ok=True)
    parser_code_path.write_bytes(Path(ingest_forwards.__file__).read_bytes())
    parser_code_hash = hashlib.sha256(parser_code_path.read_bytes()).hexdigest()
    parser_config = {
        "parser_version": "ingest_forwards.v1",
        "selection_mode": "latest_available",
        "markets": {"CH": source_date.date().isoformat()},
    }
    parser_config_hash = capstone._sha256_json(parser_config)
    legacy_receipt_unsigned: dict[str, object] = {
        "schema_version": TRUSTED_TIME_RECEIPT_SCHEMA,
        "source_document_id": workbook_path.name,
        "source_document_sha256": workbook_hash,
        "size_bytes": len(workbook_payload),
        "received_at_utc": observed.isoformat(),
        "journal_id": os.environ["PFC_DATA_TIMESTAMP_JOURNAL_ID"],
        "journal_sequence": 1,
        "previous_receipt_id": "",
    }
    legacy_receipt_unsigned["receipt_id"] = capstone._sha256_json(legacy_receipt_unsigned)
    legacy_receipt = sign_eex_trusted_time_receipt(
        legacy_receipt_unsigned,
        private_key_path=os.environ["PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH"],
    )
    vintage_row: dict[str, object] = {
        "date": source_date,
        "observed_at": observed,
        "available_at": observed,
        "acquisition_id": legacy_receipt_unsigned["receipt_id"],
        "product": product,
        "load_type": "BASE",
        "product_type": "CAL",
        "price": price,
        "unit": "EUR/MWH",
        "market": "CH",
        "source": "EEX",
        "source_document_id": workbook_path.name,
        "source_document_sha256": workbook_hash,
        "source_sheet": "CH",
        "source_row_index": 4,
        "source_column_index": 2,
        "source_product_code": f"Y01_{product}_BASE",
        "parser_version": "ingest_forwards.v1",
        "parser_code_sha256": parser_code_hash,
        "parser_config_sha256": parser_config_hash,
        "revision_sequence": 1,
        "supersedes_quote_id": "",
        "revision_timestamp": observed,
        "ingestion_run_id": f"fixture-{source_date.date()}",
        "schema_version": EEX_HISTORICAL_QUOTE_SCHEMA,
    }
    vintage_row["snapshot_id"] = historical_vintage_snapshot_id(vintage_row)
    vintage_row["revision_id"] = historical_vintage_revision_id(vintage_row)
    vintage_row["row_hash"] = historical_vintage_row_hash(vintage_row)
    vintage_row["quote_id"] = vintage_row["row_hash"]
    vintage_frame = pd.DataFrame([vintage_row])
    vintage_frame.to_parquet(source_path, index=False)
    payload = source_path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    catalog_unsigned: dict[str, object] = {
        "schema_version": EEX_HISTORICAL_CATALOG_SCHEMA,
        "created_at_utc": max(reference, observed).isoformat(),
        "data_cutoff_utc": observed.isoformat(),
        "calibration_eligible": True,
        "source_class": "IMMUTABLE_DAILY_EEX_XLSX",
        "revision_policy": "APPEND_ONLY_PRESERVE_ALL_REVISIONS",
        "ompex_used": False,
        "history_parquet": {
            "path": "history.parquet",
            "sha256": source_hash,
            "size_bytes": len(payload),
            "row_count": len(vintage_frame),
        },
        "source_documents": [
            {
                "snapshot_ids": [vintage_row["snapshot_id"]],
                "acquisition_id": vintage_row["acquisition_id"],
                "observed_at": observed.isoformat(),
                "available_at": observed.isoformat(),
                "source_document_id": workbook_path.name,
                "path": f"sources/{workbook_path.name}",
                "sha256": workbook_hash,
                "size_bytes": len(workbook_payload),
                "parser_code_path": "parser/ingest_forwards.py",
                "parser_code_sha256": parser_code_hash,
                "parser_config": parser_config,
                "parser_config_sha256": parser_config_hash,
                "trusted_time_receipt": legacy_receipt,
            }
        ],
    }
    catalog_unsigned["catalog_id"] = capstone._sha256_json(catalog_unsigned)
    catalog = sign_acquisition_contract(
        catalog_unsigned,
        private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
    )
    catalog_path = catalog_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    catalog_payload = catalog_path.read_bytes()
    parser_config_path = catalog_root / "parser" / "config.json"
    parser_config_path.write_text(json.dumps(parser_config), encoding="utf-8")
    quality_path = catalog_root / "quality.json"
    quality_policy_hash = "e" * 64
    quality_path.write_text(
        json.dumps(
            {
                "schema_version": "lt_source_quality.v1",
                "role": role,
                "status": "PASS",
                "policy_sha256": quality_policy_hash,
            }
        ),
        encoding="utf-8",
    )
    role_available = max(reference, observed)

    def mutate(contract: dict[str, object]) -> None:
        files = contract["files"]
        assert isinstance(files, dict)
        existing_receipts = sorted(
            (entry["source_receipt"] for entry in files.values()),
            key=lambda receipt: int(receipt["sequence"]),
        )
        last_receipt = existing_receipts[-1]
        source_receipt = sign_source_acquisition_receipt(
            {
                "schema_version": "source_acquisition_receipt.v2",
                "journal_id": os.environ["PFC_DATA_TIMESTAMP_JOURNAL_ID"],
                "sequence": int(last_receipt["sequence"]) + 1,
                "previous_receipt_id": last_receipt["receipt_id"],
                "acquisition_id": contract["acquisition_id"],
                "source_role": role,
                "source_system": "EEX_MARKET_DATA",
                "source_locator": "fixture://direct-eex-workbook",
                "acquisition_method": "PROSPECTIVE_DIRECT",
                "received_at_utc": role_available.isoformat(),
                "raw_sha256": workbook_hash,
                "raw_size_bytes": len(workbook_payload),
            },
            private_key_path=os.environ["PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH"],
        )
        files[role] = {
            "path": logical_path,
            "size_bytes": len(payload),
            "sha256": source_hash,
            "acquisition_id": contract["acquisition_id"],
            "available_at_utc": role_available.isoformat(),
            "calibration_eligible": True,
            "expected_cadence_seconds": None,
            "source_system": "EEX_MARKET_DATA",
            "source_receipt": source_receipt,
            "raw_artifact": {
                "path": workbook_path.relative_to(generation_root).as_posix(),
                "size_bytes": len(workbook_payload),
                "sha256": workbook_hash,
            },
            "derivation": {
                "parser_code_path": parser_code_path.relative_to(generation_root).as_posix(),
                "parser_code_sha256": parser_code_hash,
                "parser_config_path": parser_config_path.relative_to(generation_root).as_posix(),
                "parser_config_sha256": hashlib.sha256(parser_config_path.read_bytes()).hexdigest(),
                "derived_at_utc": role_available.isoformat(),
            },
            "quality_evidence": {
                "status": "PASS",
                "policy_sha256": quality_policy_hash,
                "report_path": quality_path.relative_to(generation_root).as_posix(),
                "report_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
            },
            "vintage_catalog": {
                "path": catalog_path.relative_to(generation_root).as_posix(),
                "sha256": hashlib.sha256(catalog_payload).hexdigest(),
                "size_bytes": len(catalog_payload),
            },
        }
        contract["available_at_utc"] = role_available.isoformat()
        receipt_ids = [
            str(entry["source_receipt"]["receipt_id"])
            for entry in sorted(
                files.values(),
                key=lambda entry: int(entry["source_receipt"]["sequence"]),
            )
        ]
        contract["source_journal_checkpoint"] = sign_source_journal_checkpoint(
            {
                "schema_version": "source_journal_checkpoint.v1",
                "journal_id": os.environ["PFC_DATA_TIMESTAMP_JOURNAL_ID"],
                "acquisition_id": contract["acquisition_id"],
                "first_sequence": 1,
                "last_sequence": len(receipt_ids),
                "previous_receipt_id": None,
                "receipt_ids": receipt_ids,
                "head_receipt_id": receipt_ids[-1],
                "issued_at_utc": role_available.isoformat(),
                "bundle_root_sha256": governed_snapshot_bundle_root_sha256(files),
            },
            private_key_path=os.environ["PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH"],
        )

    _replace_candidate_data_contract(paths, mutate=mutate, resign=True)
    manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    manifest["input_sources"][role] = {
        "role": role,
        "logical_path": logical_path,
        "size_bytes": len(payload),
        "sha256": source_hash,
        "rows": len(vintage_frame),
        "frame_sha256": dataframe_sha256(vintage_frame),
    }
    config_path = (
        paths["candidate_run_manifest"].parent.parent / "evidence" / "config" / "config.yaml"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["forwards"]["monthly_curve_solver"]["enabled"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    paths["candidate_run_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    production = json.loads(paths["production_manifest"].read_text(encoding="utf-8"))
    production["monthly_level_authority"] = "solver"
    selected_history = materialize_eex_forward_history_as_of(
        vintage_frame,
        valuation_timestamp=role_available,
    )
    summary_reference = max(reference, pd.to_datetime(selected_history["date"], utc=True).max())
    validated = capstone.validate_governed_forward_history_frame(
        selected_history,
        reference_timestamp=summary_reference,
    )
    source_summary = list(validated.attrs["eex_source_summary"])
    production["source_hashes"] = {
        "eex_forwards_history": source_hash,
        "eex_forwards_history_source_summary": capstone._sha256_json(source_summary),
        "eex_historical_vintage_catalog": hashlib.sha256(catalog_payload).hexdigest(),
        "eex_forwards_history_consumed_frame": dataframe_sha256(validated),
    }
    production["eex_forwards_history_source_summary"] = source_summary
    production["verified_vintage_catalog"] = {
        "catalog_id": catalog_unsigned["catalog_id"],
        "catalog_sha256": hashlib.sha256(catalog_payload).hexdigest(),
        "history_sha256": source_hash,
        "snapshot_count": 1,
        "source_document_count": 1,
        "data_cutoff_utc": observed.isoformat(),
        "status": "VERIFIED_SIGNED_IMMUTABLE_VINTAGES",
        "pit_selection": dict(selected_history.attrs["pit_selection"]),
    }
    paths["production_manifest"].write_text(json.dumps(production), encoding="utf-8")
    selected = json.loads(paths["selected_config"].read_text(encoding="utf-8"))
    selected["candidate_manifest_sha256"] = hashlib.sha256(
        paths["production_manifest"].read_bytes()
    ).hexdigest()
    paths["selected_config"].write_text(json.dumps(selected), encoding="utf-8")
    _write_candidate_bundle_manifest(
        paths["candidate_bundle_manifest"].parent,
        paths["candidate_bundle_manifest"],
    )


def _governed_history_evidence() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-16"]),
            "product": ["2027"],
            "load_type": ["BASE"],
            "product_type": ["CAL"],
            "price": [80.0],
            "market": ["CH"],
            "source": ["EEX"],
        }
    )
    validated = capstone.validate_governed_forward_history_frame(
        history,
        reference_timestamp=pd.Timestamp("2026-06-17T00:00:00Z"),
    )
    validated.attrs["verified_vintage_catalog"] = {
        "catalog_id": "c" * 64,
        "catalog_sha256": "b" * 64,
        "history_sha256": "a" * 64,
        "snapshot_count": 1,
        "source_document_count": 1,
        "data_cutoff_utc": "2026-06-16T12:00:00+00:00",
        "status": "VERIFIED_SIGNED_IMMUTABLE_VINTAGES",
        "pit_selection": {
            "valuation_timestamp": "2026-06-17T00:00:00+00:00",
            "selected_rows": 1,
            "eligible_vintage_rows": 1,
            "max_available_at": "2026-06-16T12:00:00+00:00",
        },
    }
    return validated, list(validated.attrs["eex_source_summary"])


def _replace_candidate_data_contract(
    paths: dict[str, Path],
    *,
    mutate,
    resign: bool,
) -> None:
    archived_contract = (
        paths["candidate_run_manifest"].parent.parent
        / "evidence"
        / "data"
        / "lt_input_snapshot.json"
    )
    contract = json.loads(archived_contract.read_text(encoding="utf-8"))
    contract.pop("acquisition_attestation", None)
    mutate(contract)
    if resign:
        contract = sign_acquisition_contract(
            contract,
            private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
        )
    raw = json.dumps(contract).encode("utf-8")
    archived_contract.write_bytes(raw)
    live_contract = (
        paths["data_root"] / "snapshots" / "governed-generation-001" / "lt_input_snapshot.json"
    )
    live_contract.write_bytes(raw)
    _refresh_external_publication_evidence(paths)
    _write_candidate_bundle_manifest(
        paths["candidate_bundle_manifest"].parent,
        paths["candidate_bundle_manifest"],
    )


def _refresh_external_publication_evidence(paths: dict[str, Path]) -> None:
    bundle = paths["candidate_run_manifest"].parent.parent
    data_dir = bundle / "evidence" / "data"
    contract_path = data_dir / "lt_input_snapshot.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    generation_id = str(contract["generation_id"])
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    publication = _external_publication_projection(
        paths["data_root"],
        generation_id=generation_id,
        contract_sha256=contract_hash,
    )
    pointer_path = data_dir / "current.json"
    intent_path = data_dir / "snapshot_publication_intent.json"
    receipt_path = data_dir / "snapshot_anchor_receipt.json"
    observation_path = data_dir / "snapshot_anchor_head_observation.json"
    pointer_path.write_bytes(publication["pointer"])
    intent_path.write_bytes(publication["intent"])
    receipt_path.write_bytes(publication["receipt"])
    observation_path.write_bytes(publication["observation"])

    run_manifest = json.loads(paths["candidate_run_manifest"].read_text(encoding="utf-8"))
    run_manifest["data_contract"]["sha256"] = contract_hash
    for key, path in (
        ("data_pointer", pointer_path),
        ("data_publication_intent", intent_path),
        ("data_publication_anchor_receipt", receipt_path),
        ("data_publication_head_observation", observation_path),
    ):
        run_manifest[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    paths["candidate_run_manifest"].write_text(json.dumps(run_manifest), encoding="utf-8")


def _audit_gates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"gate_id": "hard_monthly_curve_repricing", "status": "PASS", "product": "2028"},
            {"gate_id": "neighbor_level_leakage", "status": "PASS", "product": "neighbor_shift"},
            {
                "gate_id": "monthly_shape_regression_2028_2030",
                "status": "PASS",
                "product": "2028_2030_focus_population",
            },
        ]
    )


def _manifest(
    *,
    monthly_solution_hash: str,
    forward_binding: tuple[dict[str, object], dict[str, object]] | None = None,
) -> dict[str, object]:
    snapshot, eligibility = forward_binding or (
        {
            "snapshot_id": "snapshot-id",
            "quote_set_sha256": "quote-set-sha256",
            "quote_lineage_sha256": "quote-lineage-sha256",
        },
        {
            "eligibility_id": "eligibility-id",
            "snapshot_id": "snapshot-id",
            "status": "PASS",
            "requested_role": "HARD_LEVEL",
        },
    )
    hierarchy_policy = {
        "schema_version": "monthly_quote_hierarchy_policy.v1",
        "priority": ["MONTH", "QUARTER", "CALENDAR"],
        "quote_conflict_tolerance_eur_mwh": 0.01,
        "hard_repricing_tolerance_eur_mwh": 1e-9,
        "stationarity_tolerance": 1e-7,
        "residual_lineage_required": True,
    }
    hierarchy_hash = hashlib.sha256(
        json.dumps(hierarchy_policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    hard_quotes = [
        dict(quote)
        for quote in snapshot.get("quotes", [])
        if isinstance(quote, dict) and quote.get("load_type") == "BASE"
    ]
    provenance_rows: list[dict[str, object]] = []
    for quote in hard_quotes:
        lineage_payload = {
            "formula": "direct_parent_mean",
            "parent_product": quote["product"],
            "parent_quote_id": quote["quote_id"],
            "source_quote_ids": [quote["quote_id"]],
            "known_buckets": [],
            "target": float(quote["price_eur_mwh"]),
        }
        lineage_hash = hashlib.sha256(
            json.dumps(lineage_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        provenance_rows.append(
            {
                "constraint_name": f"BASE:{quote['product']}",
                "product": quote["product"],
                "parent_product": quote["product"],
                "source_quote_ids": quote["quote_id"],
                "lineage_formula": "direct_parent_mean",
                "lineage_sha256": lineage_hash,
                "lineage_payload": json.dumps(
                    lineage_payload, sort_keys=True, separators=(",", ":")
                ),
                "target": float(quote["price_eur_mwh"]),
                "hours": float(count_hours(2027, 1, 1, country="CH")[0]),
                "n_months": 1,
                "is_residual": False,
                "load_type": "BASE",
            }
        )
    hard_quote_hash = hashlib.sha256(
        json.dumps(hard_quotes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provenance_hash = hashlib.sha256(
        json.dumps(provenance_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    delivery_months = sorted(
        {
            str(period)
            for quote in hard_quotes
            for period in capstone.product_periods(str(quote["product"]))
        }
    )
    return {
        "delivery_months": delivery_months or ["2027-01"],
        "monthly_level_authority": "legacy",
        "monthly_solution_hash": monthly_solution_hash,
        "active_constraints_hash": "constraints",
        "forward_snapshot": snapshot,
        "forward_eligibility": eligibility,
        "product_hierarchy_policy": hierarchy_policy,
        "product_hierarchy_policy_sha256": hierarchy_hash,
        "hard_quotes": hard_quotes,
        "hard_quote_set_hash": hard_quote_hash,
        "constraint_provenance_rows": provenance_rows,
        "constraint_provenance_hash": provenance_hash,
        "solver_config": {
            "lambda_prior": 1e-6,
            "lambda_smooth_month": 1.0,
            "lambda_smooth_yoy": 0.25,
            "lambda_shape": 1.0,
            "neighbor_shrinkage": 0.5,
            "history_lookback_years": 6,
            "min_history_snapshots": 24,
            "constraint_tolerance": 1e-9,
            "stationarity_tolerance": 1e-7,
        },
        "solver_kkt": {
            "max_abs_constraint_residual": 1e-12,
            "stationarity_residual": 1e-12,
            "condition_number": 100.0,
            "ridge_used": False,
            "solved_by_lstsq": False,
        },
    }


def _write_delivered_product_fixture(
    csv_path: Path,
    forwards_path: Path,
    snapshot: dict[str, object],
) -> None:
    quotes = [dict(quote) for quote in snapshot["quotes"]]
    normalized_quotes: list[dict[str, object]] = []
    for quote in quotes:
        product = str(quote["product"])
        load_type = str(quote["load_type"]).upper()
        if load_type == "PEAK" and product.lower().endswith("-peak"):
            product = product[:-5]
        normalized_quotes.append(
            {
                "date": pd.Timestamp(snapshot["snapshot_date"]).tz_localize(None),
                "product": product,
                "load_type": load_type,
                "product_type": "Month",
                "price": float(quote["price_eur_mwh"]),
                "market": "CH",
            }
        )
    pd.DataFrame(normalized_quotes).to_parquet(forwards_path, index=False)
    base = next(float(row["price"]) for row in normalized_quotes if row["load_type"] == "BASE")
    peak = next(float(row["price"]) for row in normalized_quotes if row["load_type"] == "PEAK")
    timestamps = pd.date_range(
        "2027-01-01T00:00:00",
        "2027-02-01T00:00:00",
        freq="1h",
        inclusive="left",
        tz="Europe/Zurich",
    )
    peak_mask = eex_peak_mask(pd.Series(timestamps), country="CH").to_numpy()
    offpeak = (base * len(timestamps) - peak * int(peak_mask.sum())) / int((~peak_mask).sum())
    delivered = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "price_weighted_mean_eur_mwh": [peak if flag else offpeak for flag in peak_mask],
        }
    )
    delivered.to_csv(csv_path, index=False)


def _forward_binding_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    report = tmp_path / "eex_report.xlsx"
    rows = [
        [None, "M01_2027_BASE", "M01_2027_PEAK"],
        [None, "ISIN-BASE", "ISIN-PEAK"],
        ["Date", None, None],
        ["17.06.2026", 81.0, 100.0],
    ]
    with pd.ExcelWriter(report, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="CH", index=False, header=False)
    report_bytes = report.read_bytes()
    prices, snapshot_date, metadata = load_base_prices_from_eex_report_bytes(
        report_bytes,
        market="CH",
        source_label=str(report.resolve()),
        return_snapshot_metadata=True,
    )
    snapshot = ForwardSnapshot(
        prices=prices,
        market="CH",
        source_kind="EEX_XLSX",
        source_description="capstone identity fixture",
        snapshot_date=snapshot_date,
        available_at="2026-06-17T00:00:00Z",
        source_path=str(report.resolve()),
        source_sha256=hashlib.sha256(report_bytes).hexdigest(),
        quote_lineage=metadata["quote_lineage"],
        source_sheet="CH",
    )
    snapshot_manifest = snapshot.to_manifest()
    snapshot_manifest["hard_quote_eligible"] = True
    eligibility = ForwardEligibility(
        snapshot_id=snapshot.snapshot_id,
        observation_id=snapshot.observation_id,
        source_sha256=str(snapshot.source_sha256),
        quote_set_sha256=snapshot.quote_set_sha256,
        quote_lineage_sha256=snapshot.quote_lineage_sha256,
        reference_timestamp=pd.Timestamp("2026-06-17T00:00:00Z"),
        available_at=pd.Timestamp("2026-06-17T00:00:00Z"),
        business_age_days=0,
        max_age_business_days=1,
    )
    return snapshot_manifest, eligibility.to_manifest()


def _product_normalization_summary() -> dict[str, object]:
    return {
        "schema_version": "ch_product_normalization_audit.v3",
        "all_gates_pass": True,
        "covered_hard_gates_pass": True,
        "supported_hard_gates_no_critical": True,
        "critical_count": 0,
        "unsupported_count": 0,
        "quote_conflict_count": 0,
        "blocking_quote_conflict_count": 0,
        "accepted_quote_conflict_count": 0,
        "delivered_curve_drift_count": 0,
        "input_csv_sha256": "input-csv-sha256",
        "forwards_sha256": "forwards-sha256",
        "forward_snapshot_date": "2026-06-17",
        "market": "CH",
        "gate_id_counts": {
            "hard_base_product_repricing": 4,
            "hard_peak_product_repricing": 4,
            "implied_offpeak_identity": 4,
        },
        "status_counts": {"PASS": 12},
        "source_hierarchy_policy": {
            "status": "NOT_PROVIDED",
            "production_approved": False,
            "blocking_quote_conflict_count": 0,
        },
    }
