from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.pipeline.atomic_promotion as atomic_promotion
import pfc_shaping.pipeline.governed_release as governed_release
import scripts.check_monthly_curve_promotion_from_manifests as capstone
from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    build_monthly_constraint_system,
)
from pfc_shaping.data import ingest_entso, ingest_epex, ingest_forwards, ingest_hydro
from pfc_shaping.data.acquisition_contract import (
    GOVERNED_LT_INPUT_SNAPSHOT_SCHEMA,
    PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
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
from pfc_shaping.data.forward_proxy import (
    load_forward_snapshot,
    validate_forward_snapshot,
)
from pfc_shaping.data.governed_lt_acquisition import (
    ENERGY_CHARTS_PRICE_URL,
    ENERGY_CHARTS_SOURCE_SYSTEM,
    ENTSOE_API_URL,
    ENTSOE_SOURCE_SYSTEM,
    SFOE_OGD17_URL,
    SFOE_SOURCE_SYSTEM,
    CapturedProviderDocument,
    approved_provider_parser_path_for_role,
    build_provider_transform_config,
    build_raw_envelope,
    transform_provider_raw,
)
from pfc_shaping.data.lt_input_replay import (
    approved_parser_path_for_role,
    build_replay_config,
)
from pfc_shaping.data.lt_input_sources import (
    dataframe_sha256,
    resolve_lt_input_paths,
    validate_governed_forward_history_frame,
)
from pfc_shaping.data.snapshot_anchor_signing import (
    sign_snapshot_anchor_head_observation,
    sign_snapshot_anchor_receipt,
)
from pfc_shaping.data.snapshot_publication_contract import (
    PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
    PUBLICATION_DOMAIN_ID_ENV,
    PUBLICATION_HEAD_CHALLENGE_NONCE_ENV,
    PUBLICATION_HEAD_OBSERVATION_PATH_ENV,
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
from pfc_shaping.pipeline.atomic_promotion import (
    PromotionError,
    finalize_assembled_candidate_staging,
    verify_candidate_bundle,
)
from pfc_shaping.pipeline.candidate_evidence import (
    CandidateEvidenceError,
    _verified_eex_acquisition,
    capture_pre_run_governance_evidence,
    seal_assembled_candidate_evidence,
    verify_assembled_candidate_evidence,
)
from pfc_shaping.pipeline.candidate_evidence_assembler import (
    ASSEMBLY_MANIFEST,
    HOURLY_EXPORT,
    CandidateEvidenceAssemblyError,
    _canonical_frame_records,
    _delivery_quarter_hour_grid,
    _hash_frame,
    _sha256_json,
    assemble_candidate_derived_evidence,
    verify_candidate_derived_evidence,
)
from pfc_shaping.pipeline.candidate_product_evidence import (
    CONFLICT_INVENTORY,
    STAGED_POLICY,
    CandidateProductEvidenceError,
    assemble_candidate_product_evidence,
    verify_candidate_product_evidence,
)
from pfc_shaping.pipeline.governed_release import (
    GovernedReleaseError,
    audit_release_request,
    register_assembled_release_request,
)
from pfc_shaping.pipeline.model_governance_contract import (
    sign_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    active_monthly_curve_config_payload,
    monthly_solver_settings,
)
from pfc_shaping.pipeline.quality_gate import validate_input_frame
from pfc_shaping.pipeline.quote_conflict_policy_contract import (
    sign_quote_conflict_policy,
)
from pfc_shaping.pipeline.release_request_contract import ensure_workflow_domain
from scripts.assemble_lt_candidate_product_evidence import main as product_evidence_main
from scripts.finalize_lt_candidate import main as finalize_candidate_main

_SOURCE_SYSTEM_BY_ROLE = {
    "epex_ch": ENERGY_CHARTS_SOURCE_SYSTEM,
    "epex_de": ENERGY_CHARTS_SOURCE_SYSTEM,
    "entso": ENTSOE_SOURCE_SYSTEM,
    "hydro": SFOE_SOURCE_SYSTEM,
    "eex_forwards_history": "EEX_MARKET_DATA",
}


def _provision_registration_roots(
    tmp_path: Path,
    *,
    workflow_name: str,
    evidence_name: str,
) -> tuple[Path, Path]:
    workflow = tmp_path / workflow_name
    evidence = tmp_path / evidence_name
    workflow.mkdir()
    evidence.mkdir()
    ensure_workflow_domain(workflow, create=True)
    return workflow, evidence


@pytest.fixture(autouse=True)
def _trusted_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        capstone,
        "verify_external_operation_receipt",
        lambda payload: payload,
    )
    monkeypatch.setenv(
        "PFC_PROMOTION_RELEASE_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174003",
    )
    monkeypatch.setenv(
        "PFC_RELEASE_WORKFLOW_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174004",
    )
    for prefix, private_env, public_env in (
        (
            "model",
            "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
            "PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "data",
            "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
            "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "data-time",
            "PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH",
            "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "data-journal",
            "PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH",
            "PFC_DATA_JOURNAL_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "data-publication-request",
            PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV,
            PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV,
        ),
        (
            "data-publication-anchor",
            "PFC_TEST_DATA_PUBLICATION_ANCHOR_PRIVATE_KEY_PATH",
            PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
        ),
        (
            "quote-policy",
            "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
            "PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "release-request",
            "PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
            "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH",
        ),
    ):
        key = Ed25519PrivateKey.generate()
        private = tmp_path / f"{prefix}-private.pem"
        public = tmp_path / f"{prefix}-public.pem"
        private.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        public.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setenv(private_env, str(private))
        monkeypatch.setenv(public_env, str(public))
    for index, public_env in enumerate(
        (
            "PFC_DATA_PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_BASE_MODEL_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_SELECTION_INPUT_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_TIMESTAMP_JOURNAL_WITNESS_TRUSTED_PUBLIC_KEY_PATH",
        )
    ):
        key = Ed25519PrivateKey.generate()
        public = tmp_path / f"global-trust-role-{index}.pem"
        public.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setenv(public_env, str(public))
    monkeypatch.setenv("PFC_DATA_TIMESTAMP_JOURNAL_ID", "candidate-source-journal")
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "2bc0bb59-88a6-4da9-b7a4-dcbea91ae36f",
    )


def _write_external_publication_projection(
    data_root: Path,
    *,
    generation_id: str,
    contract_sha256: str,
    operation_id: str = "00000000-0000-4000-8000-000000000101",
) -> dict[str, bytes | str]:
    intent = sign_snapshot_publication_intent(
        {
            "schema_version": "lt_snapshot_publication_intent.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "operation_id": operation_id,
            "operation_created_at_utc": "2026-07-13T23:59:20+00:00",
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
            "committed_at_utc": "2026-07-13T23:59:25+00:00",
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
            "observed_at_utc": "2026-07-13T23:59:30+00:00",
            "expires_at_utc": "2026-07-14T00:04:30+00:00",
        },
        private_key_path=os.environ["PFC_TEST_DATA_PUBLICATION_ANCHOR_PRIVATE_KEY_PATH"],
    )
    observation_payload = canonical_json(observation)

    view = data_root / "views" / "pfc_lt"
    intent_path = view / "intents" / f"{intent['intent_id']}.json"
    receipt_path = view / "receipts" / f"{receipt['receipt_id']}.json"
    observation_path = view / "observations" / f"{observation['observation_id']}.json"
    for path, payload in (
        (intent_path, intent_payload),
        (receipt_path, receipt_payload),
        (observation_path, observation_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (view / "current.json").write_bytes(pointer_payload)
    os.environ[PUBLICATION_HEAD_OBSERVATION_PATH_ENV] = str(observation_path)
    os.environ[PUBLICATION_HEAD_CHALLENGE_NONCE_ENV] = challenge_nonce
    return {
        "pointer": pointer_payload,
        "intent": intent_payload,
        "receipt": receipt_payload,
        "observation": observation_payload,
        "challenge_nonce": challenge_nonce,
    }


def _sign_governed_input_contract(
    files: dict[str, object],
    *,
    generation_id: str,
    acquisition_id: str,
    available_at_utc: str,
    schema_version: str = GOVERNED_LT_INPUT_SNAPSHOT_SCHEMA,
) -> dict[str, object]:
    enriched: dict[str, object] = {}
    previous: str | None = None
    timestamp_private = os.environ["PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH"]
    for sequence, (role, raw_entry) in enumerate(sorted(files.items()), start=1):
        entry = dict(raw_entry)
        source_system = str(entry["source_system"])
        receipt_artifact = dict(
            entry.get("provider_raw_artifact")
            if schema_version == PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA
            and role in {"epex_ch", "epex_de", "entso", "hydro"}
            else entry.get("raw_artifact")
            or {
                "path": f"raw/{role}.parquet",
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
        )
        receipt = sign_source_acquisition_receipt(
            {
                "schema_version": "source_acquisition_receipt.v2",
                "journal_id": "candidate-source-journal",
                "sequence": sequence,
                "previous_receipt_id": previous,
                "acquisition_id": acquisition_id,
                "source_role": role,
                "source_system": source_system,
                "source_locator": str(
                    entry.get(
                        "source_locator",
                        {
                            "epex_ch": ENERGY_CHARTS_PRICE_URL,
                            "epex_de": ENERGY_CHARTS_PRICE_URL,
                            "entso": ENTSOE_API_URL,
                            "hydro": SFOE_OGD17_URL,
                        }.get(role, f"fixture://{role}"),
                    )
                ),
                "acquisition_method": "PROSPECTIVE_DIRECT",
                "received_at_utc": entry["available_at_utc"],
                "raw_sha256": receipt_artifact["sha256"],
                "raw_size_bytes": receipt_artifact["size_bytes"],
            },
            private_key_path=timestamp_private,
        )
        previous = str(receipt["receipt_id"])
        entry["source_receipt"] = receipt
        entry.setdefault("raw_artifact", receipt_artifact)
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
            "journal_id": "candidate-source-journal",
            "acquisition_id": acquisition_id,
            "first_sequence": 1,
            "last_sequence": len(receipt_ids),
            "previous_receipt_id": None,
            "receipt_ids": receipt_ids,
            "head_receipt_id": receipt_ids[-1],
            "issued_at_utc": available_at_utc,
            "bundle_root_sha256": governed_snapshot_bundle_root_sha256(enriched),
        },
        private_key_path=os.environ["PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH"],
    )
    return sign_acquisition_contract(
        {
            "schema_version": schema_version,
            "layout": "external_v2",
            "generation_id": generation_id,
            "acquisition_id": acquisition_id,
            "available_at_utc": available_at_utc,
            "calibration_eligible": True,
            "source_class": "GOVERNED_ACQUISITION",
            "source_journal_checkpoint": checkpoint,
            "files": enriched,
        },
        private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
    )


def _write_historical_thresholds(path: Path) -> None:
    columns = [
        "gate_id",
        "metric",
        "market",
        "delivery_bucket",
        "lookback_start",
        "lookback_end",
        "n_snapshots",
        "min_required_n",
        "p50",
        "p90",
        "p975",
        "max_observed",
        "regime_filter",
        "status",
    ]
    rows = [",".join(columns)]
    for gate, metric in (
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        (
            "residual_vs_implied_comparable_block",
            "comparable_block_shape_delta_abs_eur_mwh",
        ),
    ):
        for bucket in ["all", *(f"month_{month:02d}" for month in range(1, 13))]:
            rows.append(f"{gate},{metric},CH,{bucket},,,0,24,,,,,fixture,UNSUPPORTED")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_governed_eex_vintage(
    generation: Path,
    *,
    history: pd.DataFrame,
    reference: pd.Timestamp,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    role = "eex_forwards_history"
    catalog_root = generation / "eex-vintages"
    history_path = catalog_root / "history.parquet"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    source_date = pd.Timestamp(history.iloc[0]["date"]).tz_localize(None).normalize()
    observed = source_date.tz_localize("UTC") + pd.Timedelta(hours=18)
    product = str(history.iloc[0]["product"])
    price = float(history.iloc[0]["price"])

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
    parser_config_path = catalog_root / "parser" / "config.json"
    parser_config_path.write_text(json.dumps(parser_config), encoding="utf-8")
    parser_config_hash = _sha256_json(parser_config)

    receipt_unsigned: dict[str, object] = {
        "schema_version": TRUSTED_TIME_RECEIPT_SCHEMA,
        "source_document_id": workbook_path.name,
        "source_document_sha256": workbook_hash,
        "size_bytes": len(workbook_payload),
        "received_at_utc": observed.isoformat(),
        "journal_id": os.environ["PFC_DATA_TIMESTAMP_JOURNAL_ID"],
        "journal_sequence": 1,
        "previous_receipt_id": "",
    }
    receipt_unsigned["receipt_id"] = _sha256_json(receipt_unsigned)
    trusted_time_receipt = sign_eex_trusted_time_receipt(
        receipt_unsigned,
        private_key_path=os.environ["PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH"],
    )

    vintage_row: dict[str, object] = {
        "date": source_date,
        "observed_at": observed,
        "available_at": observed,
        "acquisition_id": receipt_unsigned["receipt_id"],
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
    vintage_frame.to_parquet(history_path, index=False)
    history_payload = history_path.read_bytes()
    history_hash = hashlib.sha256(history_payload).hexdigest()

    catalog_unsigned: dict[str, object] = {
        "schema_version": EEX_HISTORICAL_CATALOG_SCHEMA,
        "created_at_utc": observed.isoformat(),
        "data_cutoff_utc": observed.isoformat(),
        "calibration_eligible": True,
        "source_class": "IMMUTABLE_DAILY_EEX_XLSX",
        "revision_policy": "APPEND_ONLY_PRESERVE_ALL_REVISIONS",
        "ompex_used": False,
        "history_parquet": {
            "path": "history.parquet",
            "sha256": history_hash,
            "size_bytes": len(history_payload),
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
                "trusted_time_receipt": trusted_time_receipt,
            }
        ],
    }
    catalog_unsigned["catalog_id"] = _sha256_json(catalog_unsigned)
    catalog = sign_acquisition_contract(
        catalog_unsigned,
        private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
    )
    catalog_path = catalog_root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    catalog_payload = catalog_path.read_bytes()
    catalog_hash = hashlib.sha256(catalog_payload).hexdigest()

    quality_policy_hash = "e" * 64
    quality_path = catalog_root / "quality.json"
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
    selected = materialize_eex_forward_history_as_of(
        vintage_frame,
        valuation_timestamp=reference,
    )
    validated = validate_governed_forward_history_frame(
        selected,
        reference_timestamp=reference,
    )
    source_summary = list(validated.attrs["eex_source_summary"])
    role_entry: dict[str, object] = {
        "path": history_path.relative_to(generation).as_posix(),
        "size_bytes": len(history_payload),
        "sha256": history_hash,
        "acquisition_id": "input-001",
        "available_at_utc": observed.isoformat(),
        "calibration_eligible": True,
        "expected_cadence_seconds": None,
        "source_system": _SOURCE_SYSTEM_BY_ROLE[role],
        "raw_artifact": {
            "path": workbook_path.relative_to(generation).as_posix(),
            "size_bytes": len(workbook_payload),
            "sha256": workbook_hash,
        },
        "derivation": {
            "parser_code_path": parser_code_path.relative_to(generation).as_posix(),
            "parser_code_sha256": parser_code_hash,
            "parser_config_path": parser_config_path.relative_to(generation).as_posix(),
            "parser_config_sha256": hashlib.sha256(parser_config_path.read_bytes()).hexdigest(),
            "derived_at_utc": observed.isoformat(),
        },
        "quality_evidence": {
            "status": "PASS",
            "policy_sha256": quality_policy_hash,
            "report_path": quality_path.relative_to(generation).as_posix(),
            "report_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        },
        "vintage_catalog": {
            "path": catalog_path.relative_to(generation).as_posix(),
            "sha256": catalog_hash,
            "size_bytes": len(catalog_payload),
        },
    }
    input_receipt = {
        "role": role,
        "logical_path": role_entry["path"],
        "size_bytes": len(history_payload),
        "sha256": history_hash,
        "rows": len(vintage_frame),
        "frame_sha256": dataframe_sha256(vintage_frame),
    }
    evidence = {
        "history_sha256": history_hash,
        "catalog_sha256": catalog_hash,
        "source_summary": source_summary,
        "consumed_frame_sha256": dataframe_sha256(validated),
        "verified_vintage_catalog": {
            "catalog_id": catalog_unsigned["catalog_id"],
            "catalog_sha256": catalog_hash,
            "history_sha256": history_hash,
            "snapshot_count": 1,
            "source_document_count": 1,
            "data_cutoff_utc": observed.isoformat(),
            "status": "VERIFIED_SIGNED_IMMUTABLE_VINTAGES",
            "pit_selection": dict(selected.attrs["pit_selection"]),
        },
    }
    return role_entry, input_receipt, evidence


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=True)
    return buffer.getvalue()


def _frame_quality_metrics(frame: pd.DataFrame) -> dict[str, object]:
    metrics = {
        "row_count": len(frame),
        "start_utc": frame.index.min().tz_convert("UTC").isoformat().replace("+00:00", "Z"),
        "end_utc": frame.index.max().tz_convert("UTC").isoformat().replace("+00:00", "Z"),
        "missing_value_count": int(frame.isna().sum().sum()),
        "duplicate_timestamp_count": int(frame.index.duplicated().sum()),
    }
    if "lt_observation_resolution_provenance" in frame.attrs:
        metrics["resolution_provenance"] = dict(
            frame.attrs["lt_observation_resolution_provenance"]
        )
    return metrics


def _entsoe_fixture_xml(frame: pd.DataFrame, *, generation: bool) -> bytes:
    def series(column: str, psr_type: str | None = None) -> str:
        psr = f"<MktPSRType><psrType>{psr_type}</psrType></MktPSRType>" if psr_type else ""
        points = "".join(
            f"<Point><position>{position}</position><quantity>{value}</quantity></Point>"
            for position, value in enumerate(frame[column].tolist(), start=1)
        )
        start = frame.index.min().strftime("%Y-%m-%dT%H:%MZ")
        end = (frame.index.max() + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%MZ")
        return (
            f"<TimeSeries><mRID>{column}-series</mRID>"
            f"<businessType>{'A01' if generation else 'A04'}</businessType>"
            f"<objectAggregation>{'A08' if generation else 'A01'}</objectAggregation>"
            "<curveType>A01</curveType>"
            f"<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>{psr}"
            f"<Period><timeInterval><start>{start}</start>"
            f"<end>{end}</end></timeInterval><resolution>PT15M</resolution>"
            f"{points}</Period></TimeSeries>"
        )

    payload = (
        series("solar_mw", "B16") + series("wind_mw", "B19") if generation else series("load_mw")
    )
    if generation:
        namespace = "urn:iec62325.351:tc57wg16:451-6:generation-document:3:0"
        identity = (
            "<type>A75</type><process.processType>A16</process.processType>"
            "<in_Domain.mRID>10YCH-SWISSGRIDZ</in_Domain.mRID>"
        )
    else:
        namespace = "urn:iec62325.351:tc57wg16:451-6:load-document:3:0"
        identity = (
            "<type>A65</type><process.processType>A16</process.processType>"
            "<outBiddingZone_Domain.mRID>10YCH-SWISSGRIDZ"
            "</outBiddingZone_Domain.mRID>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<GL_MarketDocument xmlns="{namespace}">{identity}{payload}'
        "</GL_MarketDocument>"
    ).encode("ascii")


def _provider_envelope(
    role: str,
    frame: pd.DataFrame,
    *,
    received_at_utc: str,
) -> tuple[bytes, str, str, str]:
    start = frame.index.min().tz_convert("UTC")
    end = (
        frame.index.max() + pd.Timedelta(days=1)
        if role == "hydro"
        else frame.index.max() + pd.Timedelta(minutes=15)
    ).tz_convert("UTC")
    start_utc = start.isoformat().replace("+00:00", "Z")
    end_utc = end.isoformat().replace("+00:00", "Z")
    if role.startswith("epex_"):
        zone = {"epex_ch": "CH", "epex_de": "DE-LU"}[role]
        body = json.dumps(
            {
                "deprecated": False,
                "license_info": "CC BY 4.0 test fixture",
                "unix_seconds": [int(value.timestamp()) for value in frame.index],
                "price": frame["price_eur_mwh"].tolist(),
                "unit": "EUR / MWh",
            },
            separators=(",", ":"),
        ).encode("ascii")
        documents = [
            CapturedProviderDocument(
                document_id="day_ahead_price",
                request_url=ENERGY_CHARTS_PRICE_URL,
                request_parameters={
                    "bzn": zone,
                    "start": start.strftime("%Y-%m-%dT%H:%MZ"),
                    "end": (end - pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%MZ"),
                },
                response_media_type="application/json",
                body=body,
            )
        ]
        source_system = ENERGY_CHARTS_SOURCE_SYSTEM
        source_locator = ENERGY_CHARTS_PRICE_URL
    elif role == "entso":
        parameters = {
            "processType": "A16",
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        documents = [
            CapturedProviderDocument(
                document_id="actual_load_ch",
                request_url=ENTSOE_API_URL,
                request_parameters={
                    **parameters,
                    "documentType": "A65",
                    "outBiddingZone_Domain": "10YCH-SWISSGRIDZ",
                },
                response_media_type="application/xml",
                body=_entsoe_fixture_xml(frame, generation=False),
                credential_reference="env:ENTSOE_API_KEY",
            ),
            CapturedProviderDocument(
                document_id="actual_generation_ch",
                request_url=ENTSOE_API_URL,
                request_parameters={
                    **parameters,
                    "documentType": "A75",
                    "in_Domain": "10YCH-SWISSGRIDZ",
                },
                response_media_type="application/xml",
                body=_entsoe_fixture_xml(frame, generation=True),
                credential_reference="env:ENTSOE_API_KEY",
            ),
        ]
        source_system = ENTSOE_SOURCE_SYSTEM
        source_locator = ENTSOE_API_URL
    elif role == "hydro":
        lines = [
            "Datum,Wallis_speicherinhalt_gwh,Graubuenden_speicherinhalt_gwh,"
            "Tessin_speicherinhalt_gwh,UebrigCH_speicherinhalt_gwh,"
            "TotalCH_speicherinhalt_gwh,Wallis_max_speicherinhalt_gwh,"
            "Graubuenden_max_speicherinhalt_gwh,Tessin_max_speicherinhalt_gwh,"
            "UebrigCH_max_speicherinhalt_gwh,TotalCH_max_speicherinhalt_gwh"
        ]
        lines.extend(
            f"{timestamp.tz_convert('Europe/Zurich'):%Y-%m-%d},"
            f"{row.fill_gwh * 0.4},{row.fill_gwh * 0.25},"
            f"{row.fill_gwh * 0.15},{row.fill_gwh * 0.2},{row.fill_gwh},"
            f"{row.max_capacity_gwh * 0.4},{row.max_capacity_gwh * 0.25},"
            f"{row.max_capacity_gwh * 0.15},{row.max_capacity_gwh * 0.2},"
            f"{row.max_capacity_gwh}"
            for timestamp, row in frame.iterrows()
        )
        documents = [
            CapturedProviderDocument(
                document_id="ogd17_reservoir_levels",
                request_url=SFOE_OGD17_URL,
                request_parameters={},
                response_media_type="text/csv",
                body=("\n".join(lines) + "\n").encode("ascii"),
            )
        ]
        source_system = SFOE_SOURCE_SYSTEM
        source_locator = SFOE_OGD17_URL
    else:
        raise AssertionError(role)
    return (
        build_raw_envelope(
            acquisition_id="input-001",
            source_role=role,
            source_system=source_system,
            source_locator=source_locator,
            provider_id=source_system,
            received_at_utc=received_at_utc,
            documents=documents,
        ),
        start_utc,
        end_utc,
        source_locator,
    )


def _governed_input_generation(
    root: Path,
    *,
    reference: pd.Timestamp,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    generation = root / "snapshots" / "input-001"
    generation.mkdir(parents=True)
    actual_end = reference.floor("15min") - pd.Timedelta(minutes=15)
    common_index = pd.date_range(
        actual_end - pd.Timedelta(days=7),
        actual_end,
        freq="15min",
    )
    hydro_index = pd.date_range(
        end=reference.tz_convert("Europe/Zurich").normalize(),
        periods=312,
        freq="W-MON",
    ).tz_convert("UTC")
    hydro_fill_pct = [
        42.0 + float(position % 52) * 0.2 + float(position // 52) * 0.5
        for position in range(len(hydro_index))
    ]
    bronze_frames = {
        "epex_ch": pd.DataFrame({"price_eur_mwh": 80.0}, index=common_index),
        "epex_de": pd.DataFrame({"price_eur_mwh": 75.0}, index=common_index),
        "entso": pd.DataFrame(
            {
                "load_mw": [7000.0 + float(position % 96) for position in range(len(common_index))],
                "solar_mw": 1000.0,
                "wind_mw": 500.0,
            },
            index=common_index,
        ),
        "hydro": pd.DataFrame(
            {
                "fill_pct": hydro_fill_pct,
                "fill_gwh": [value * 100.0 for value in hydro_fill_pct],
                "max_capacity_gwh": 10000.0,
                "uebrig_ch_gwh": [value * 20.0 for value in hydro_fill_pct],
                "wallis_gwh": [value * 40.0 for value in hydro_fill_pct],
                "graubuenden_gwh": [value * 25.0 for value in hydro_fill_pct],
                "tessin_gwh": [value * 15.0 for value in hydro_fill_pct],
            },
            index=hydro_index,
        ),
    }
    for role in ("epex_ch", "epex_de"):
        count = len(bronze_frames[role])
        bronze_frames[role].attrs["lt_observation_resolution_provenance"] = {
            "schema_version": "lt_observation_resolution_provenance.v2",
            "role": role,
            "source_cadence_seconds": 900,
            "output_cadence_seconds": 900,
            "source_observation_count": count,
            "output_observation_count": count,
            "resampling_method": "NONE",
            "output_sampling_kind": "DIRECT_15_MINUTE_OBSERVATIONS_PRODUCT_IDENTITY_UNVERIFIED",
            "native_quarter_hour_cadence_eligible": True,
            "native_hourly_cadence_eligible": False,
            "hourly_aggregation_eligible": True,
            "native_quarter_hour_truth_eligible": False,
            "product_identity_status": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_UNVERIFIED",
            "quarter_hour_truth_blocker": "EXTERNAL_PRODUCT_AUCTION_SESSION_IDENTITY_REQUIRED",
            "scientific_use_class": "DIRECT_15_MINUTE_OBSERVATIONS_PRODUCT_IDENTITY_UNVERIFIED",
        }
    frames = {
        "epex_ch": ingest_epex._clean(bronze_frames["epex_ch"]),
        "epex_de": ingest_epex._clean(bronze_frames["epex_de"]),
        "entso": ingest_entso.build_features(bronze_frames["entso"]),
        "hydro": ingest_hydro.build_water_value(bronze_frames["hydro"]),
    }
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-13"]),
            "product": ["2027"],
            "load_type": ["BASE"],
            "product_type": ["CAL"],
            "price": [82.0],
            "market": ["CH"],
            "source": ["EEX"],
        }
    )
    files: dict[str, object] = {}
    receipts: dict[str, object] = {}
    reports: dict[str, object] = {}
    for role, frame in frames.items():
        available_at_utc = "2026-07-13T23:50:00Z"
        provider_payload, provider_start, provider_end, _ = _provider_envelope(
            role,
            bronze_frames[role],
            received_at_utc=available_at_utc,
        )
        provider_raw_path = generation / "provider_raw" / f"{role}.json"
        provider_raw_path.parent.mkdir(parents=True, exist_ok=True)
        provider_raw_path.write_bytes(provider_payload)
        provider_parser_source = approved_provider_parser_path_for_role(role)
        provider_parser_payload = provider_parser_source.read_bytes()
        provider_parser_path = generation / "provider_parser" / f"{role}.py"
        provider_parser_path.parent.mkdir(parents=True, exist_ok=True)
        provider_parser_path.write_bytes(provider_parser_payload)
        provider_config = build_provider_transform_config(
            role=role,
            start_utc=provider_start,
            end_utc=provider_end,
        )
        provider_config_payload = json.dumps(
            provider_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        provider_config_path = generation / "provider_parser" / f"{role}.json"
        provider_config_path.write_bytes(provider_config_payload)
        pd.testing.assert_frame_equal(
            transform_provider_raw(
                envelope_payload=provider_payload,
                parser_config=provider_config,
            ),
            bronze_frames[role],
            check_freq=False,
        )
        logical = f"inputs/{role}.parquet"
        path = generation / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _parquet_payload(frame)
        path.write_bytes(payload)
        raw_path = generation / "raw" / f"{role}.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = _parquet_payload(bronze_frames[role])
        raw_path.write_bytes(raw_payload)
        parser_code_path = generation / "parser" / f"{role}.py"
        parser_code_path.parent.mkdir(parents=True, exist_ok=True)
        parser_code_path.write_bytes(approved_parser_path_for_role(role).read_bytes())
        parser_config_path = generation / "parser" / f"{role}.json"
        parser_config_path.write_text(
            json.dumps(
                build_replay_config(
                    role=role,
                    source_system=_SOURCE_SYSTEM_BY_ROLE[role],
                    raw_frame=bronze_frames[role],
                    derived_frame=frame,
                )
            ),
            encoding="utf-8",
        )
        quality_policy_hash = "c" * 64
        quality_path = generation / "quality" / f"{role}.json"
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_payload = (
            json.dumps(
                {
                    "bindings": {
                        "provider_raw_sha256": hashlib.sha256(provider_payload).hexdigest(),
                        "provider_parser_code_sha256": hashlib.sha256(
                            provider_parser_payload
                        ).hexdigest(),
                        "provider_parser_config_sha256": hashlib.sha256(
                            provider_config_payload
                        ).hexdigest(),
                        "bronze_sha256": hashlib.sha256(raw_payload).hexdigest(),
                        "feature_parser_code_sha256": hashlib.sha256(
                            parser_code_path.read_bytes()
                        ).hexdigest(),
                        "feature_parser_config_sha256": hashlib.sha256(
                            parser_config_path.read_bytes()
                        ).hexdigest(),
                        "derived_sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    "metrics": {
                        "bronze": _frame_quality_metrics(bronze_frames[role]),
                        "derived": _frame_quality_metrics(frame),
                    },
                    "policy_sha256": quality_policy_hash,
                    "role": role,
                    "schema_version": "lt_source_quality.v2",
                    "status": "PASS",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        quality_path.write_bytes(quality_payload)
        files[role] = {
            "path": logical,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "acquisition_id": "input-001",
            "available_at_utc": available_at_utc,
            "calibration_eligible": True,
            "expected_cadence_seconds": 604800 if role == "hydro" else 900,
            "source_system": _SOURCE_SYSTEM_BY_ROLE[role],
            "provider_raw_artifact": {
                "path": provider_raw_path.relative_to(generation).as_posix(),
                "size_bytes": len(provider_payload),
                "sha256": hashlib.sha256(provider_payload).hexdigest(),
            },
            "provider_derivation": {
                "parser_code_path": provider_parser_path.relative_to(generation).as_posix(),
                "parser_code_sha256": hashlib.sha256(provider_parser_payload).hexdigest(),
                "parser_config_path": provider_config_path.relative_to(generation).as_posix(),
                "parser_config_sha256": hashlib.sha256(provider_config_payload).hexdigest(),
                "derived_at_utc": available_at_utc,
            },
            "raw_artifact": {
                "path": raw_path.relative_to(generation).as_posix(),
                "size_bytes": len(raw_payload),
                "sha256": hashlib.sha256(raw_payload).hexdigest(),
            },
            "derivation": {
                "parser_code_path": parser_code_path.relative_to(generation).as_posix(),
                "parser_code_sha256": hashlib.sha256(parser_code_path.read_bytes()).hexdigest(),
                "parser_config_path": parser_config_path.relative_to(generation).as_posix(),
                "parser_config_sha256": hashlib.sha256(parser_config_path.read_bytes()).hexdigest(),
                "derived_at_utc": available_at_utc,
            },
            "quality_evidence": {
                "status": "PASS",
                "policy_sha256": quality_policy_hash,
                "report_path": quality_path.relative_to(generation).as_posix(),
                "report_sha256": hashlib.sha256(quality_payload).hexdigest(),
            },
        }
        receipts[role] = {
            "role": role,
            "logical_path": logical,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(frame),
            "frame_sha256": dataframe_sha256(frame),
        }
        columns = {
            "epex_ch": ["price_eur_mwh"],
            "epex_de": ["price_eur_mwh"],
            "entso": ["load_mw", "solar_mw", "wind_mw"],
            "hydro": ["fill_deviation", "water_value_supported"],
        }[role]
        report = validate_input_frame(
            frame,
            name=role,
            required_columns=columns,
            min_rows=1,
            max_age_days=10.0 if role == "hydro" else 3.0 if role == "entso" else 2.0,
            reference_timestamp=reference,
            fail_on_stale=True,
            min_finite_fraction=0.95,
            recent_window_days=7.0,
            min_recent_finite_fraction=0.95,
            max_finite_gap_days=(
                14.0 if role == "hydro" else 2.0 / 24.0 if role == "entso" else 1.0 / 24.0
            ),
        )
        reports[role] = {
            "dataset": report.dataset,
            "checks_passed": report.checks_passed,
            "warnings": report.warnings,
            "errors": report.errors,
            "metrics": report.metrics,
        }
    eex_entry, eex_receipt, eex_evidence = _write_governed_eex_vintage(
        generation,
        history=history,
        reference=reference,
    )
    files["eex_forwards_history"] = eex_entry
    receipts["eex_forwards_history"] = eex_receipt
    return files, receipts, reports, eex_evidence


def _fixture(
    tmp_path: Path,
    *,
    production_hash: str | None = None,
    staging_path: Path | None = None,
    governed_data_root: Path | None = None,
    extra_config: dict[str, object] | None = None,
) -> Path:
    staging = staging_path or (tmp_path / "candidate")
    staging.mkdir(parents=True)
    authority = tmp_path / "authority"
    authority.mkdir()
    config: dict[str, object] = {
        "forwards": {
            "monthly_curve_solver": {
                "enabled": True,
                "target_markets": ["CH"],
            }
        },
        "quality": {
            "benchmark_policy": "advisory",
            "fail_on_benchmark": False,
        },
    }
    config.update(extra_config or {})
    canonical = active_monthly_curve_config_payload(monthly_solver_settings(config))
    active_hash = config_hash(canonical)
    thresholds_source = authority / "historical_thresholds.csv"
    selected_source = authority / "selected.json"
    _write_historical_thresholds(thresholds_source)
    selected_source.write_text(
        json.dumps(
            {
                "schema_version": "monthly_curve_selected_lambda_decision.v1",
                "decision_id": "decision-test",
                "calibration_campaign_id": "campaign-test",
                "selection_reason": "governed fixture",
                "selection_status": "PRODUCTION_APPROVED",
                "production_approved": True,
                "canonical_config": canonical,
                "config_hash": active_hash,
            }
        ),
        encoding="utf-8",
    )
    receipts: dict[str, Path] = {}
    for role, artifact in (
        ("historical_thresholds", thresholds_source),
        ("selected_lambda_decision", selected_source),
    ):
        receipt = authority / f"{role}.receipt.json"
        receipt.write_text(
            json.dumps(
                sign_model_governance_artifact_receipt(
                    artifact,
                    artifact_role=role,
                    authority_id="FMV_MODEL_GOVERNANCE_TEST",
                    issued_at_utc="2026-07-12T00:00:00Z",
                    data_cutoff_utc="2026-07-11T00:00:00Z",
                    private_key_path=os.environ["PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"],
                )
            ),
            encoding="utf-8",
        )
        receipts[role] = receipt
    eex_source = authority / "Price_Report_EEX.xlsx"
    workbook_rows = [
        [
            None,
            "M01_2027_BASE",
            "M02_2027_BASE",
            "M03_2027_BASE",
            "M01_2027_PEAK",
            "M02_2027_PEAK",
            "M03_2027_PEAK",
        ],
        [None, "ISIN-1", "ISIN-2", "ISIN-3", "ISIN-4", "ISIN-5", "ISIN-6"],
        ["Date", None, None, None, None, None, None],
        ["13.07.2026", 83.5, 82.0, 81.0, 83.5, 82.0, 81.0],
    ]
    with pd.ExcelWriter(eex_source, engine="openpyxl") as writer:
        pd.DataFrame(workbook_rows).to_excel(
            writer,
            sheet_name="CH",
            index=False,
            header=False,
        )
    eex_contract = authority / "eex-acquisition.json"
    eex_contract.write_text(
        json.dumps(
            _sign_governed_input_contract(
                {
                    "eex_forward_source": {
                        "path": eex_source.name,
                        "size_bytes": eex_source.stat().st_size,
                        "sha256": hashlib.sha256(eex_source.read_bytes()).hexdigest(),
                        "acquisition_id": "eex-001",
                        "available_at_utc": "2026-07-13T18:00:00+00:00",
                        "calibration_eligible": True,
                        "source_system": "EEX_MARKET_DATA",
                    }
                },
                generation_id="eex-001",
                acquisition_id="eex-001",
                available_at_utc="2026-07-13T18:00:00+00:00",
            )
        ),
        encoding="utf-8",
    )
    runtime_config = authority / "runtime-config.yaml"
    runtime_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    (staging / "markets" / "CH").mkdir(parents=True)
    (staging / "evidence" / "config").mkdir(parents=True)
    (staging / "evidence" / "data").mkdir(parents=True)
    (staging / "evidence" / "eex").mkdir(parents=True)

    config_path = staging / "evidence" / "config" / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    snapshot = staging / "evidence" / "data" / "lt_input_snapshot.json"
    if governed_data_root is None:
        input_files = {
            role: {
                "path": f"curated/{role}.parquet",
                "size_bytes": len(role.encode("utf-8")),
                "sha256": hashlib.sha256(role.encode("utf-8")).hexdigest(),
                "acquisition_id": "input-001",
                "available_at_utc": "2026-07-12T18:00:00+00:00",
                "calibration_eligible": True,
                "source_system": _SOURCE_SYSTEM_BY_ROLE[role],
            }
            for role in ("epex_ch", "epex_de", "entso", "hydro", "eex_forwards_history")
        }
        for role in ("epex_ch", "epex_de", "entso", "hydro"):
            entry = input_files[role]
            provider_payload = f"provider-envelope:{role}".encode("ascii")
            bronze_payload = f"bronze:{role}".encode("ascii")
            entry["provider_raw_artifact"] = {
                "path": f"provider_raw/{role}.json",
                "size_bytes": len(provider_payload),
                "sha256": hashlib.sha256(provider_payload).hexdigest(),
            }
            entry["provider_derivation"] = {
                "parser_code_path": f"provider_parser/{role}.py",
                "parser_code_sha256": hashlib.sha256(
                    f"provider-parser:{role}".encode("ascii")
                ).hexdigest(),
                "parser_config_path": f"provider_parser/{role}.json",
                "parser_config_sha256": hashlib.sha256(
                    f"provider-config:{role}".encode("ascii")
                ).hexdigest(),
                "derived_at_utc": entry["available_at_utc"],
            }
            entry["raw_artifact"] = {
                "path": f"raw/{role}.parquet",
                "size_bytes": len(bronze_payload),
                "sha256": hashlib.sha256(bronze_payload).hexdigest(),
            }
        input_receipts = {
            role: {
                "role": role,
                "logical_path": entry["path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
                "rows": 1,
                "frame_sha256": hashlib.sha256(f"frame:{role}".encode("utf-8")).hexdigest(),
            }
            for role, entry in input_files.items()
        }
        freshness_reports: dict[str, object] = {}
        eex_evidence: dict[str, object] = {}
    else:
        (
            input_files,
            input_receipts,
            freshness_reports,
            eex_evidence,
        ) = _governed_input_generation(
            governed_data_root,
            reference=pd.Timestamp("2026-07-13T23:59:59Z"),
        )
    snapshot.write_text(
        json.dumps(
            _sign_governed_input_contract(
                input_files,
                generation_id="input-001",
                acquisition_id="input-001",
                available_at_utc=(
                    "2026-07-13T23:50:00+00:00"
                    if governed_data_root is not None
                    else "2026-07-12T18:00:00+00:00"
                ),
                schema_version=PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
            )
        ),
        encoding="utf-8",
    )
    publication_root = governed_data_root or (tmp_path / "publication-data")
    live_contract = publication_root / "snapshots" / "input-001" / "lt_input_snapshot.json"
    live_contract.parent.mkdir(parents=True, exist_ok=True)
    live_contract.write_bytes(snapshot.read_bytes())
    publication = _write_external_publication_projection(
        publication_root,
        generation_id="input-001",
        contract_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    )
    pointer = staging / "evidence" / "data" / "current.json"
    pointer.write_bytes(publication["pointer"])
    publication_intent = staging / "evidence" / "data" / "snapshot_publication_intent.json"
    publication_receipt = staging / "evidence" / "data" / "snapshot_anchor_receipt.json"
    publication_observation = (
        staging / "evidence" / "data" / "snapshot_anchor_head_observation.json"
    )
    publication_intent.write_bytes(publication["intent"])
    publication_receipt.write_bytes(publication["receipt"])
    publication_observation.write_bytes(publication["observation"])
    pre_run_capture = tmp_path / "pre-run-capture"
    pre_run_capture.mkdir()
    capture_pre_run_governance_evidence(
        pre_run_capture,
        run_id="run-001",
        valuation_timestamp="2026-07-13T23:59:59+00:00",
        build_timestamp="2026-07-14T00:00:00+00:00",
        historical_thresholds_path=thresholds_source,
        historical_thresholds_receipt_path=receipts["historical_thresholds"],
        selected_lambda_decision_path=selected_source,
        selected_lambda_decision_receipt_path=receipts["selected_lambda_decision"],
        eex_forward_source_path=eex_source,
        eex_acquisition_contract_path=eex_contract,
        runtime_config_path=runtime_config,
        expected_runtime_config_sha256=hashlib.sha256(runtime_config.read_bytes()).hexdigest(),
        source_revision="1" * 40,
        input_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        input_pointer_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(),
        input_generation_id="input-001",
        publication_head_observation_sha256=hashlib.sha256(
            publication_observation.read_bytes()
        ).hexdigest(),
        publication_head_challenge_nonce=str(publication["challenge_nonce"]),
        peak_source_policy="same_first",
        use_seasonal_hourly_shape=True,
    )
    shutil.copytree(pre_run_capture, staging, dirs_exist_ok=True)
    pre_run_path = staging / "manifests" / "pre_run_governance_manifest.json"
    source = staging / "evidence" / "eex" / "source.xlsx"
    source.write_bytes(eex_source.read_bytes())
    spot = pd.DataFrame(
        {"price_eur_mwh": [80.0]},
        index=pd.date_range("2026-07-01", periods=1, freq="15min", tz="UTC"),
    )
    with patch(
        "pfc_shaping.data.forward_proxy._observation_timestamp_utc",
        return_value=pd.Timestamp("2026-07-13T18:00:00Z"),
    ):
        forward_snapshot = load_forward_snapshot(
            spot,
            eex_report_path=str(eex_source),
            config={},
            market="CH",
            allow_spot_proxy=False,
        )
    forward_eligibility = validate_forward_snapshot(
        forward_snapshot,
        reference_timestamp="2026-07-13T23:59:59+00:00",
    )
    quotes = staging / "evidence" / "eex" / "forward_quote_snapshot.parquet"
    forward_manifest = forward_snapshot.to_manifest()
    forward_manifest["source_path"] = "../evidence/eex/source.xlsx"
    forward_manifest["source_description"] = "archived EEX source in test candidate bundle"
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-13"),
                "product": str(quote["product"]).removesuffix("-Peak").removesuffix("-Offpeak"),
                "load_type": str(quote["load_type"]),
                "product_type": "Month",
                "price": float(quote["price_eur_mwh"]),
                "market": "CH",
                "source": "candidate_forward_snapshot",
                "quote_id": str(quote["quote_id"]),
            }
            for quote in forward_manifest["quotes"]
        ]
    ).to_parquet(quotes, index=False)
    delivery_months = pd.period_range("2027-01", "2027-03", freq="M")
    pfc_index = pd.date_range(
        pd.Timestamp("2027-01-01", tz="Europe/Zurich").tz_convert("UTC"),
        pd.Timestamp("2027-04-01", tz="Europe/Zurich").tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )
    local_month = pfc_index.tz_convert("Europe/Zurich").month
    month_prices = {1: 83.5, 2: 82.0, 3: 81.0}
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pd.DataFrame(
        {"price_shape": [month_prices[int(month)] for month in local_month]},
        index=pfc_index,
    ).to_parquet(pfc_path)
    solver_quotes = [
        MarketQuote(
            market="CH",
            product=str(quote["product"]),
            load_type="BASE",
            price=float(quote["price_eur_mwh"]),
            snapshot_date=pd.Timestamp(forward_manifest["snapshot_date"]),
            source=str(forward_manifest["source_kind"]),
            available_at=pd.Timestamp(forward_manifest["available_at"]),
            quote_id=str(quote["quote_id"]),
            snapshot_id=str(forward_manifest["snapshot_id"]),
            observation_id=str(forward_manifest["observation_id"]),
            source_kind=str(forward_manifest["source_kind"]),
            source_sha256=str(forward_manifest["source_sha256"]),
        )
        for quote in forward_manifest["quotes"]
        if str(quote["load_type"]).upper() == "BASE"
    ]
    constraints = build_monthly_constraint_system(
        delivery_months,
        solver_quotes,
        market="CH",
        load_type="BASE",
    )
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
    constraint_provenance = _canonical_frame_records(constraints.rows[provenance_columns])
    hard_quotes = [
        dict(quote)
        for quote in forward_manifest["quotes"]
        if str(quote["load_type"]).upper() == "BASE"
    ]
    solver_settings = monthly_solver_settings(config)
    product_policy = {
        "schema_version": "monthly_quote_hierarchy_policy.v1",
        "priority": ["MONTH", "QUARTER", "CALENDAR"],
        "quote_conflict_tolerance_eur_mwh": float(solver_settings["quote_conflict_tolerance"]),
        "hard_repricing_tolerance_eur_mwh": float(solver_settings["constraint_tolerance"]),
        "stationarity_tolerance": float(solver_settings["stationarity_tolerance"]),
        "residual_lineage_required": True,
    }
    production = staging / "manifests" / "production_monthly_curve_manifest_ch.json"
    monthly_solution = {"2027-01": 83.5, "2027-02": 82.0, "2027-03": 81.0}
    monthly_solution_hash = _sha256_json(monthly_solution)
    production.write_text(
        json.dumps(
            {
                "market": "CH",
                "monthly_level_authority": "solver",
                "solver_config": solver_settings,
                "solver_config_hash": _sha256_json(solver_settings),
                "solver_kkt": {
                    "max_abs_constraint_residual": 1e-12,
                    "stationarity_residual": 1e-12,
                    "condition_number": 100.0,
                    "ridge_used": False,
                    "solved_by_lstsq": False,
                },
                "monthly_solution_hash": monthly_solution_hash,
                "monthly_solution": monthly_solution,
                "active_constraints_hash": _hash_frame(constraints.rows),
                "quote_diagnostics_hash": _hash_frame(constraints.quote_diagnostics),
                "constraint_provenance_rows": constraint_provenance,
                "constraint_provenance_hash": _sha256_json(constraint_provenance),
                "hard_quotes": hard_quotes,
                "hard_quote_set_hash": _sha256_json(hard_quotes),
                "active_config_hash": production_hash or active_hash,
                "promotion_eligible": True,
                "delivery_months": [str(month) for month in delivery_months],
                "forward_snapshot": forward_manifest,
                "forward_eligibility": forward_eligibility.to_manifest(),
                "product_hierarchy_policy": product_policy,
                "product_hierarchy_policy_sha256": _sha256_json(product_policy),
            }
        ),
        encoding="utf-8",
    )
    if governed_data_root is not None:
        production_payload = json.loads(production.read_text(encoding="utf-8"))
        history_summary = list(eex_evidence["source_summary"])
        production_payload["source_hashes"] = {
            "eex_forwards_history": eex_evidence["history_sha256"],
            "eex_forwards_history_source_summary": _sha256_json(history_summary),
            "eex_historical_vintage_catalog": eex_evidence["catalog_sha256"],
            "eex_forwards_history_consumed_frame": eex_evidence["consumed_frame_sha256"],
        }
        production_payload["eex_forwards_history_source_summary"] = history_summary
        production_payload["verified_vintage_catalog"] = eex_evidence["verified_vintage_catalog"]
        production.write_text(json.dumps(production_payload), encoding="utf-8")
    run_manifest = {
        "schema_version": "lt_candidate_run.v1",
        "run_id": "run-001",
        "pipeline_scope": "LT_ONLY",
        "promotion_eligible": False,
        "probabilistic_status": "DETERMINISTIC_ONLY",
        "interval_columns": [],
        "intervals_permitted_for_pricing_or_risk": False,
        "candidate_serialized_at_utc": "2026-07-13T23:59:59+00:00",
        "run_started_at_utc": "2026-07-13T23:59:59+00:00",
        "reference_timestamp": "2026-07-13T23:59:59+00:00",
        "reference_timestamp_is_explicit": True,
        "config_path": config_path.relative_to(staging).as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "pre_run_governance_manifest": pre_run_path.relative_to(staging).as_posix(),
        "pre_run_governance_manifest_sha256": hashlib.sha256(pre_run_path.read_bytes()).hexdigest(),
        "data_contract": {
            "archive_path": snapshot.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        },
        "data_pointer": {
            "archive_path": pointer.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
            "logical_path": "views/pfc_lt/current.json",
        },
        "data_publication_intent": {
            "archive_path": publication_intent.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(publication_intent.read_bytes()).hexdigest(),
        },
        "data_publication_anchor_receipt": {
            "archive_path": publication_receipt.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(publication_receipt.read_bytes()).hexdigest(),
        },
        "data_publication_head_observation": {
            "archive_path": publication_observation.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(publication_observation.read_bytes()).hexdigest(),
        },
        "data_generation_id": "input-001",
        "input_sources": input_receipts,
        "source_evidence": {
            "forward_source_archive": source.relative_to(staging).as_posix(),
            "forward_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "forward_quote_snapshot": quotes.relative_to(staging).as_posix(),
            "forward_quote_snapshot_sha256": hashlib.sha256(quotes.read_bytes()).hexdigest(),
            "forward_snapshot_id": forward_snapshot.snapshot_id,
            "forward_observation_id": forward_snapshot.observation_id,
        },
        "markets": {
            "CH": {
                "rows": int(len(pfc_index)),
                "columns": list(pd.read_parquet(pfc_path).columns),
                "probabilistic_status": "DETERMINISTIC_ONLY",
                "pfc_parquet": pfc_path.relative_to(staging).as_posix(),
                "pfc_parquet_sha256": hashlib.sha256(pfc_path.read_bytes()).hexdigest(),
                "monthly_curve_manifest": production.relative_to(staging).as_posix(),
                "monthly_curve_manifest_sha256": hashlib.sha256(
                    production.read_bytes()
                ).hexdigest(),
            }
        },
    }
    deterministic_artifacts = {
        path.relative_to(staging).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root_name in ("markets", "models")
        if (staging / root_name).is_dir()
        for path in sorted((staging / root_name).rglob("*"))
        if path.is_file()
    }
    run_manifest["deterministic_artifacts"] = deterministic_artifacts
    run_manifest["deterministic_artifacts_sha256"] = hashlib.sha256(
        json.dumps(
            deterministic_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if governed_data_root is not None:
        run_manifest["data_layout"] = "external_v2"
        run_manifest["data_root_id"] = "fmv_data_view:pfc_lt"
        run_manifest["freshness_reports"] = freshness_reports
    (staging / "manifests" / "candidate_run_manifest.json").write_text(
        json.dumps(run_manifest),
        encoding="utf-8",
    )
    return staging


def test_assembly_replays_hourly_export_from_staged_pfc(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)

    manifest = assemble_candidate_derived_evidence(staging, run_id="run-001")

    assert manifest["schema_version"] == "lt_candidate_derived_evidence_assembly.v1"
    assert set(manifest["sources"]) == {
        "active_config",
        "input_snapshot_manifest",
        "input_snapshot_pointer",
        "forward_source",
        "eex_acquisition_contract",
        "forward_quote_snapshot",
        "production_manifest",
        "publication_anchor_receipt",
        "publication_head_observation",
        "publication_intent",
        "pfc_ch_15min",
        "selected_lambda_decision",
        "historical_thresholds",
    }
    hourly = pd.read_csv(staging / HOURLY_EXPORT)
    assert (
        len(hourly) == len(pd.read_parquet(staging / "markets" / "CH" / "pfc_15min.parquet")) // 4
    )
    assert hourly["price_weighted_mean_eur_mwh"].iloc[0] == 83.5
    monthly_gates = pd.read_csv(staging / "audits" / "monthly_curve_gates.csv")
    assert not monthly_gates.empty
    assert not monthly_gates["status"].eq("CRITICAL").any()
    assert verify_candidate_derived_evidence(staging, expected_run_id="run-001") == manifest


def test_assembly_rejects_incomplete_global_authority_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = _fixture(tmp_path)
    monkeypatch.delenv("PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH")

    with pytest.raises(
        CandidateEvidenceAssemblyError,
        match="candidate external publication evidence is invalid",
    ):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_interval_columns_even_if_candidate_hash_is_updated(
    tmp_path: Path,
) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path)
    pfc["p10"] = pfc["price_shape"] - 10.0
    pfc["p90"] = pfc["price_shape"] + 10.0
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["columns"] = list(pfc.columns)
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(pfc_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="interval columns"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_completed_assembly_is_idempotent(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    first = assemble_candidate_derived_evidence(staging, run_id="run-001")
    first_hash = hashlib.sha256((staging / ASSEMBLY_MANIFEST).read_bytes()).hexdigest()

    second = assemble_candidate_derived_evidence(staging, run_id="run-001")

    assert second == first
    assert hashlib.sha256((staging / ASSEMBLY_MANIFEST).read_bytes()).hexdigest() == first_hash


@pytest.mark.parametrize(
    ("month", "expected_quarter_hours"),
    [("2027-03", 2972), ("2027-10", 2980)],
)
def test_solver_delivery_grid_preserves_spring_and_autumn_dst(
    month: str,
    expected_quarter_hours: int,
) -> None:
    grid = _delivery_quarter_hour_grid(pd.PeriodIndex([month], freq="M"))

    assert len(grid) == expected_quarter_hours
    assert grid.tz is not None
    assert grid.is_unique
    assert (grid.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_verifier_rejects_hourly_export_even_after_manifest_rehash(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    hourly_path = staging / HOURLY_EXPORT
    hourly = pd.read_csv(hourly_path)
    hourly.loc[0, "price_weighted_mean_eur_mwh"] = 999.0
    hourly.to_csv(hourly_path, index=False, lineterminator="\n")
    manifest_path = staging / ASSEMBLY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["derived"]["hourly_export_ch"]["sha256"] = hashlib.sha256(
        hourly_path.read_bytes()
    ).hexdigest()
    manifest["derived"]["hourly_export_ch"]["size_bytes"] = hourly_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="replay"):
        verify_candidate_derived_evidence(staging, expected_run_id="run-001")


def test_verifier_rejects_forged_export_claims_and_dependency_graph(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    export_path = staging / "manifests" / "hourly_export_manifest_ch.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    export["market"] = "FORGED"
    export["producer_contract"] = "attacker.v1"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    manifest_path = staging / ASSEMBLY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_graph"] = {"forged": ["nothing"]}
    manifest["derived"]["export_manifest"]["sha256"] = hashlib.sha256(
        export_path.read_bytes()
    ).hexdigest()
    manifest["derived"]["export_manifest"]["size_bytes"] = export_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        CandidateEvidenceAssemblyError,
        match="dependency_graph|export_manifest.replay",
    ):
        verify_candidate_derived_evidence(staging, expected_run_id="run-001")


def test_assembly_rejects_production_manifest_config_drift(tmp_path: Path) -> None:
    staging = _fixture(tmp_path, production_hash="f" * 64)

    with pytest.raises(CandidateEvidenceAssemblyError, match="hash parity failed"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solved_by_lstsq", True),
        ("ridge_used", True),
        ("stationarity_residual", 1.0),
        ("max_abs_constraint_residual", 1.0),
        ("condition_number", "not-finite"),
        ("condition_number", 1e15),
    ],
)
def test_assembly_rejects_failed_solver_numerical_diagnostics(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    staging = _fixture(tmp_path)
    production_path = staging / "manifests" / "production_monthly_curve_manifest_ch.json"
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production["solver_kkt"][field] = value
    production_path.write_text(json.dumps(production), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["monthly_curve_manifest_sha256"] = hashlib.sha256(
        production_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="numerical diagnostics"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_pfc_monthly_level_drift(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path)
    pfc["price_shape"] += 5.0
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(pfc_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="monthly means"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_partial_solver_month_even_after_row_count_update(
    tmp_path: Path,
) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path).iloc[:-4]
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["rows"] = len(pfc)
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(pfc_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="exactly cover"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_inflated_repricing_tolerance(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    production_path = staging / "manifests" / "production_monthly_curve_manifest_ch.json"
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production["product_hierarchy_policy"]["hard_repricing_tolerance_eur_mwh"] = 6.0
    production["product_hierarchy_policy_sha256"] = _sha256_json(
        production["product_hierarchy_policy"]
    )
    production_path.write_text(json.dumps(production), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["monthly_curve_manifest_sha256"] = hashlib.sha256(
        production_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="policy mismatch"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_duplicated_forward_quote_snapshot(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    quotes_path = staging / "evidence" / "eex" / "forward_quote_snapshot.parquet"
    quotes = pd.read_parquet(quotes_path)
    pd.concat([quotes, quotes.iloc[[0]]], ignore_index=True).to_parquet(
        quotes_path,
        index=False,
    )
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["source_evidence"]["forward_quote_snapshot_sha256"] = hashlib.sha256(
        quotes_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="quote snapshot"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_misaligned_quarter_hour_grid(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path)
    pfc.index = pfc.index + pd.Timedelta(minutes=7)
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(pfc_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="quarter-hour aligned"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_missing_governed_input_snapshot(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_contract"] = None
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="input snapshot is missing"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_resolver_rejects_fully_resigned_eex_history_without_vintage_catalog(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "governed-data"
    _fixture(tmp_path, governed_data_root=data_root)
    generation = data_root / "snapshots" / "input-001"
    contract_path = generation / "lt_input_snapshot.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    files = contract["files"]
    files["eex_forwards_history"].pop("vintage_catalog")
    resigned = _sign_governed_input_contract(
        files,
        generation_id="input-001",
        acquisition_id="input-001",
        available_at_utc="2026-07-13T23:50:00+00:00",
        schema_version=PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
    )
    contract_path.write_text(json.dumps(resigned), encoding="utf-8")
    _write_external_publication_projection(
        data_root,
        generation_id="input-001",
        contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        operation_id="00000000-0000-4000-8000-000000000102",
    )

    with (
        patch(
            "pfc_shaping.data.snapshot_publication_state._reference_utc",
            return_value=datetime(2026, 7, 14, 0, 1, tzinfo=timezone.utc),
        ),
        pytest.raises(ValueError, match="requires a signed vintage_catalog binding"),
    ):
        resolve_lt_input_paths(tmp_path / "project", data_root=data_root)


def test_assembly_rejects_signed_divergent_input_acquisition_id(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    snapshot_path = staging / "evidence" / "data" / "lt_input_snapshot.json"
    contract = json.loads(snapshot_path.read_text(encoding="utf-8"))
    contract.pop("acquisition_attestation")
    contract["files"]["entso"]["acquisition_id"] = "foreign-acquisition"
    signed = sign_acquisition_contract(
        contract,
        private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
    )
    snapshot_path.write_text(json.dumps(signed), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_contract"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="authentication failed"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_signed_input_path_escape(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    snapshot_path = staging / "evidence" / "data" / "lt_input_snapshot.json"
    contract = json.loads(snapshot_path.read_text(encoding="utf-8"))
    contract.pop("acquisition_attestation")
    contract["files"]["entso"]["path"] = "../foreign/entso.parquet"
    snapshot_path.write_text(
        json.dumps(
            sign_acquisition_contract(
                contract,
                private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )
    pointer_path = staging / "evidence" / "data" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["contract_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_contract"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    run["data_pointer"]["sha256"] = hashlib.sha256(pointer_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="authentication failed"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_extra_input_receipt_role(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["input_sources"]["foreign"] = {
        "role": "foreign",
        "logical_path": "foreign.parquet",
        "size_bytes": 1,
        "sha256": "f" * 64,
        "rows": 1,
        "frame_sha256": "e" * 64,
    }
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="receipt roles are not exact"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_rehashed_foreign_pointer_generation(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    pointer_path = staging / "evidence" / "data" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_id"] = "foreign-generation"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_pointer"]["sha256"] = hashlib.sha256(pointer_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(
        CandidateEvidenceAssemblyError,
        match="external publication evidence is invalid",
    ):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_bootstrap_publication_explicitly(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    intent_path = staging / "evidence" / "data" / "snapshot_publication_intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["transition_type"] = "BOOTSTRAP"
    intent_path.write_text(json.dumps(intent), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_publication_intent"]["sha256"] = hashlib.sha256(intent_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(
        CandidateEvidenceAssemblyError,
        match="cannot consume a BOOTSTRAP publication",
    ):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_eex_acquisition_rejects_legacy_v1_contract(tmp_path: Path) -> None:
    workbook = tmp_path / "eex.xlsx"
    workbook.write_bytes(b"fixture")
    contract_path = tmp_path / "eex-contract.json"
    contract_path.write_text(
        json.dumps(
            sign_acquisition_contract(
                {
                    "schema_version": "lt_input_snapshot.v1",
                    "layout": "external_v2",
                    "generation_id": "eex-001",
                    "acquisition_id": "",
                    "calibration_eligible": True,
                    "source_class": "GOVERNED_ACQUISITION",
                    "files": {
                        "eex_forward_source": {
                            "path": workbook.name,
                            "size_bytes": workbook.stat().st_size,
                            "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                            "acquisition_id": "",
                            "available_at_utc": "2026-07-12T18:00:00+00:00",
                            "calibration_eligible": True,
                            "source_system": "EEX_MARKET_DATA",
                        }
                    },
                },
                private_key_path=os.environ["PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(CandidateEvidenceError, match="requires a governed v2"):
        _verified_eex_acquisition(
            workbook,
            contract_path,
            valuation_timestamp="2026-07-13T00:00:00+00:00",
        )


def test_renamed_ompex_source_is_rejected_by_authenticated_provider(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "generic-market-data.xlsx"
    workbook.write_bytes(b"renamed OMPEX benchmark bytes")
    contract_path = tmp_path / "eex-contract.json"
    contract_path.write_text(
        json.dumps(
            _sign_governed_input_contract(
                {
                    "eex_forward_source": {
                        "path": workbook.name,
                        "size_bytes": workbook.stat().st_size,
                        "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                        "acquisition_id": "eex-ompex-001",
                        "available_at_utc": "2026-07-12T18:00:00+00:00",
                        "calibration_eligible": True,
                        "source_system": "OMPEX_HFC",
                    }
                },
                generation_id="eex-ompex-001",
                acquisition_id="eex-ompex-001",
                available_at_utc="2026-07-12T18:00:00+00:00",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(CandidateEvidenceError, match="source system is not admissible"):
        _verified_eex_acquisition(
            workbook,
            contract_path,
            valuation_timestamp="2026-07-13T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    "section",
    ["model", "forwards", "flavors", "export", "quality"],
)
def test_assembly_rejects_ompex_marker_in_model_consumed_config(
    tmp_path: Path,
    section: str,
) -> None:
    extra = {section: {"prohibited_training_source": "generic_ompex_payload"}}
    if section == "forwards":
        extra[section]["monthly_curve_solver"] = {
            "enabled": True,
            "target_markets": ["CH"],
        }
    if section == "quality":
        extra[section].update({"benchmark_policy": "advisory", "fail_on_benchmark": False})
    staging = _fixture(tmp_path, extra_config=extra)

    with pytest.raises(CandidateEvidenceAssemblyError, match="OMPEX/HFC"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_product_evidence_requires_then_accepts_exact_signed_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")

    pending = assemble_candidate_product_evidence(staging, run_id="run-001")

    assert pending["status"] == "SOURCE_HIERARCHY_POLICY_REQUIRED"
    assert pending["promotion_eligible"] is False
    assert product_evidence_main(["--staging", str(staging), "--run-id", "run-001"]) == 2
    assert "SOURCE_HIERARCHY_POLICY_REQUIRED" in capsys.readouterr().out
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ["PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )

    evidence = assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )

    assert evidence["schema_version"] == "candidate_product_normalization_evidence.v1"
    assert (
        verify_candidate_product_evidence(
            staging,
            expected_run_id="run-001",
        )
        == evidence
    )
    summary = json.loads(
        (staging / "manifests" / "product_normalization_summary.json").read_text(encoding="utf-8")
    )
    assert summary["monthly_candidate_binding_status"] == "BOUND"
    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["all_gates_pass"] is True
    assert summary["audit_script"] == "scripts/audit_ch_product_normalization.py"
    assert not list(staging.rglob("*.tmp"))


def test_product_evidence_rejects_unsigned_policy(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "unsigned-policy.json"
    policy_path.write_text(
        json.dumps(pending["required_policy_payload"]),
        encoding="utf-8",
    )

    with pytest.raises(CandidateProductEvidenceError, match="not authentic"):
        assemble_candidate_product_evidence(
            staging,
            run_id="run-001",
            source_hierarchy_policy_path=policy_path,
        )


def test_product_inventory_recovers_exact_missing_sibling(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    first = assemble_candidate_product_evidence(staging, run_id="run-001")
    (staging / CONFLICT_INVENTORY).unlink()

    recovered = assemble_candidate_product_evidence(staging, run_id="run-001")

    assert recovered == first
    assert (staging / CONFLICT_INVENTORY).is_file()


def test_rejected_signed_policy_is_not_staged_and_can_be_retried(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    wrong_payload = dict(pending["required_policy_payload"])
    wrong_payload["input_csv_sha256"] = "f" * 64
    wrong_path = tmp_path / "wrong-signed-policy.json"
    wrong_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                wrong_payload,
                private_key_path=os.environ["PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(CandidateProductEvidenceError, match="did not pass"):
        assemble_candidate_product_evidence(
            staging,
            run_id="run-001",
            source_hierarchy_policy_path=wrong_path,
        )
    assert not (staging / STAGED_POLICY).exists()

    correct_path = tmp_path / "correct-signed-policy.json"
    correct_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ["PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )
    evidence = assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=correct_path,
    )
    assert evidence["schema_version"] == "candidate_product_normalization_evidence.v1"


def test_product_evidence_rejects_mutated_conflict_inventory(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ["PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )
    inventory_path = staging / CONFLICT_INVENTORY
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["quote_conflict_count"] = 999
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(CandidateProductEvidenceError, match="inventory replay mismatch"):
        verify_candidate_product_evidence(staging, expected_run_id="run-001")


def test_canonical_seal_and_locked_finalizer_replay_both_assemblies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    atomic_promotion._release_root_id(release, create=True)
    staging = _fixture(
        tmp_path,
        staging_path=release / "candidates" / ".run-001.staging",
    )
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ["PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )

    seal_assembled_candidate_evidence(staging, run_id="run-001")
    verify_assembled_candidate_evidence(staging, expected_run_id="run-001")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    interrupted_run = json.loads(run_path.read_text(encoding="utf-8"))
    interrupted_run.pop("candidate_evidence_manifest")
    interrupted_run.pop("candidate_evidence_manifest_sha256")
    run_path.write_text(json.dumps(interrupted_run), encoding="utf-8")
    seal_assembled_candidate_evidence(staging, run_id="run-001")
    verify_assembled_candidate_evidence(staging, expected_run_id="run-001")
    for private_key_env in (
        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(private_key_env)
    assert (
        finalize_candidate_main(
            [
                "--release-root",
                str(release),
                "--failure-root",
                str(tmp_path / "failures"),
                "--run-id",
                "run-001",
                "--source-hierarchy-policy",
                str(policy_path),
            ]
        )
        == 0
    )
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["status"] == "CANDIDATE_FINALIZED_NOT_PROMOTED"
    assert len(finalized["candidate_bundle_manifest_sha256"]) == 64
    assert (
        finalize_candidate_main(
            [
                "--release-root",
                str(release),
                "--failure-root",
                str(tmp_path / "failures"),
                "--run-id",
                "run-001",
                "--source-hierarchy-policy",
                str(policy_path),
            ]
        )
        == 0
    )
    retried = json.loads(capsys.readouterr().out)
    assert retried == finalized
    bundle = verify_candidate_bundle(release, run_id="run-001")
    recovered_bundle = finalize_assembled_candidate_staging(release, run_id="run-001")
    assert recovered_bundle.manifest_sha256 == bundle.manifest_sha256

    assert bundle.path == (release / "candidates" / "run-001").resolve()
    sealed = verify_assembled_candidate_evidence(
        bundle.path,
        expected_run_id="run-001",
    )
    assert sealed["run_id"] == "run-001"
    source = bundle.path / sealed["artifacts"]["audit_gates"]["path"]
    original_read_bytes = Path.read_bytes
    attack_workflow, attack_evidence = _provision_registration_roots(
        tmp_path,
        workflow_name="attack-workflow",
        evidence_name="attack-evidence",
    )
    monkeypatch.setenv(
        "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
        os.environ["PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH"],
    )
    with monkeypatch.context() as attack:
        attack.setattr(
            governed_release,
            "verify_candidate_bundle",
            lambda *args, **kwargs: bundle,
        )
        attack.setattr(
            governed_release,
            "verify_assembled_candidate_evidence",
            lambda *args, **kwargs: sealed,
        )

        def substitute_capture(path: Path) -> bytes:
            if path.resolve() == source.resolve():
                return b"substituted-after-seal"
            return original_read_bytes(path)

        attack.setattr(Path, "read_bytes", substitute_capture)
        with pytest.raises(GovernedReleaseError, match="role hash mismatch"):
            register_assembled_release_request(
                expected_current_event_id=None,
                release_root=release,
                workflow_root=attack_workflow,
                evidence_root=attack_evidence,
                run_id="run-001",
            )
    workflow, evidence = _provision_registration_roots(
        tmp_path,
        workflow_name="workflow",
        evidence_name="release-evidence",
    )
    with monkeypatch.context() as no_recapture:
        no_recapture.setattr(
            governed_release,
            "_capture_artifacts",
            lambda *args, **kwargs: pytest.fail(
                "assembled registration must not recapture flattened artifacts"
            ),
        )
        request = register_assembled_release_request(
            expected_current_event_id=None,
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id="run-001",
        )
    assert request["registration_contract"] == "assembled_candidate_seal.v1"
    assert set(request["artifacts"]) == {
        "audit_gates",
        "historical_thresholds",
        "production_manifest",
        "export_manifest",
        "selected_config_artifact",
        "product_normalization_summary",
    }


def test_strict_finalizer_quarantines_failed_post_rename_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    staging = _fixture(
        tmp_path,
        staging_path=release / "candidates" / ".run-001.staging",
    )
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ["PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )
    seal_assembled_candidate_evidence(staging, run_id="run-001")

    with pytest.raises(PromotionError, match="evidence_contract metadata is reserved"):
        finalize_assembled_candidate_staging(
            release,
            run_id="run-001",
            metadata={"evidence_contract": "generic_candidate.v1"},
        )
    assert staging.is_dir()
    assert not (release / "candidates" / "run-001").exists()

    original_verify = atomic_promotion.verify_assembled_candidate_evidence

    def fail_after_rename(path: str | Path, *, expected_run_id: str) -> dict[str, object]:
        if Path(path).name == "run-001":
            raise RuntimeError("injected post-rename replay failure")
        return original_verify(path, expected_run_id=expected_run_id)

    monkeypatch.setattr(
        atomic_promotion,
        "verify_assembled_candidate_evidence",
        fail_after_rename,
    )

    with pytest.raises(PromotionError, match="quarantined at"):
        finalize_assembled_candidate_staging(release, run_id="run-001")

    assert not (release / "candidates" / "run-001").exists()
    assert len(list((release / "candidates").glob(".run-001.failed-*"))) == 1


def test_assembled_release_runs_real_capstone_from_canonical_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    atomic_promotion._release_root_id(release, create=True)
    data_root = tmp_path / "governed-data"
    staging = _fixture(
        tmp_path,
        staging_path=release / "candidates" / ".run-001.staging",
        governed_data_root=data_root,
    )
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ["PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )
    seal_assembled_candidate_evidence(staging, run_id="run-001")

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 7, 14, 0, 1, tzinfo=timezone.utc)
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(atomic_promotion, "datetime", FixedDateTime)
    finalize_assembled_candidate_staging(release, run_id="run-001")
    workflow, evidence = _provision_registration_roots(
        tmp_path,
        workflow_name="workflow",
        evidence_name="release-evidence",
    )
    for private_key_env in (
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(private_key_env, raising=False)
    monkeypatch.setenv(
        "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
        os.environ["PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH"],
    )
    request = register_assembled_release_request(
        expected_current_event_id=None,
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id="run-001",
    )
    monkeypatch.delenv("PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH")
    receipt_key = Ed25519PrivateKey.generate()
    receipt_private = tmp_path / "receipt-private.pem"
    receipt_public = tmp_path / "receipt-public.pem"
    receipt_private.write_bytes(
        receipt_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    receipt_public.write_bytes(
        receipt_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(receipt_public))
    monkeypatch.delenv("PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.setattr(
        capstone,
        "_promotion_now_utc",
        lambda: pd.Timestamp("2026-07-14T00:05:00Z"),
    )

    audited = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id="run-001",
        request_id=str(request["request_id"]),
        signing_private_key=receipt_private,
    )

    assert audited["approved"] is False
    assert audited["status"] == "REJECTED"
    receipts = list(
        (workflow / "run-001" / "audit-results" / str(request["request_id"])[:32]).glob(
            "promotion_receipt.json"
        )
    )
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    governance = {row["gate_id"]: row for row in receipt["governance_gates"]}
    assert all(row["status"] == "PASS" for row in governance.values())
    assert governance["delivered_product_normalization_audit"]["status"] == "PASS"
