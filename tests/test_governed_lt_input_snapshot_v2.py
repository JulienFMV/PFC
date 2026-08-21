from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.data.snapshot_publisher as publisher_module
from pfc_shaping.data import ingest_entso, ingest_epex, ingest_hydro, snapshot_publisher
from pfc_shaping.data import snapshot_publication_state as publication_state_module
from pfc_shaping.data.acquisition_contract import (
    PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
    TRUSTED_ACQUISITION_PUBLIC_KEY_ENV,
    TRUSTED_JOURNAL_PUBLIC_KEY_ENV,
    TRUSTED_TIME_JOURNAL_ID_ENV,
    TRUSTED_TIME_PUBLIC_KEY_ENV,
    AcquisitionAuthenticationError,
    governed_snapshot_bundle_root_sha256,
    verify_acquisition_contract,
)
from pfc_shaping.data.acquisition_signing import (
    sign_acquisition_contract,
    sign_source_acquisition_receipt,
    sign_source_journal_checkpoint,
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
    validate_governed_lt_snapshot_bundle,
    validate_lt_input_contract_semantics,
)
from pfc_shaping.data.snapshot_anchor_signing import (
    sign_snapshot_anchor_head_observation,
    sign_snapshot_anchor_receipt,
)
from pfc_shaping.data.snapshot_bootstrap_authority import (
    sign_snapshot_bootstrap_authorization,
)
from pfc_shaping.data.snapshot_legacy_publication_signing import (
    sign_snapshot_publication_anchor,
    sign_snapshot_publication_event,
)
from pfc_shaping.data.snapshot_publication_contract import (
    BOOTSTRAP_REQUIRED_SOURCE_CLASS,
    BOOTSTRAP_TRANSITION_TYPE,
    BOOTSTRAP_USE_POLICY,
    PUBLICATION_ANCHOR_ROOT_ENV,
    PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
    PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
    PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_ENV,
    PUBLICATION_DOMAIN_ID_ENV,
    PUBLICATION_HEAD_CHALLENGE_NONCE_ENV,
    PUBLICATION_HEAD_OBSERVATION_PATH_ENV,
    PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV,
    PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV,
    PUBLICATION_SIGNING_PRIVATE_KEY_ENV,
    PUBLICATION_TRUSTED_PUBLIC_KEY_ENV,
    publication_domain_sha256,
    publication_private_key_id,
    publication_public_key_id,
)
from pfc_shaping.data.snapshot_publication_state import (
    SnapshotPublicationStateError,
    canonical_json,
    external_head_challenge_sha256,
    pointer_mapping_from_external_receipt,
)
from pfc_shaping.data.snapshot_publication_state import (
    _legacy_pointer_mapping_from_event_for_tests as pointer_mapping_from_event,
)
from pfc_shaping.data.snapshot_publication_state import (
    _legacy_publication_anchor_directory_for_tests as publication_anchor_directory,
)
from pfc_shaping.data.snapshot_publisher import (
    SnapshotPublicationBusy,
    SnapshotPublicationConflict,
    SnapshotPublicationRepairRequired,
    SnapshotPublicationRetired,
    finalize_governed_lt_snapshot,
    prepare_governed_lt_snapshot,
)
from pfc_shaping.data.snapshot_publisher import (
    _publish_filesystem_prototype_for_tests as publish_governed_lt_snapshot,
)
from pfc_shaping.durable_artifact import durable_artifact_durability_classification

SOURCE_SYSTEMS = {
    "epex_ch": ENERGY_CHARTS_SOURCE_SYSTEM,
    "epex_de": ENERGY_CHARTS_SOURCE_SYSTEM,
    "epex_fr": ENERGY_CHARTS_SOURCE_SYSTEM,
    "entso": ENTSOE_SOURCE_SYSTEM,
    "hydro": SFOE_SOURCE_SYSTEM,
}


def test_rejected_filesystem_publication_api_is_retired(tmp_path: Path) -> None:
    with pytest.raises(SnapshotPublicationRetired, match="filesystem publication is retired"):
        snapshot_publisher.publish_governed_lt_snapshot(
            source_bundle=tmp_path / "source",
            data_root=tmp_path / "data",
            expected_current_event_id=None,
            operation_id="00000000-0000-0000-0000-000000000001",
        )


def test_signed_migrated_v1_cannot_become_calibration_eligible(
    tmp_path: Path,
) -> None:
    private = _write_keypair(tmp_path / "acquisition", Ed25519PrivateKey.generate())[0]
    files = {
        role: {
            "path": f"inputs/{role}.parquet",
            "size_bytes": 1,
            "sha256": "a" * 64,
            "acquisition_id": "migrated-001",
            "available_at_utc": "2026-07-16T08:00:00Z",
            "calibration_eligible": True,
            "source_system": source_system,
        }
        for role, source_system in SOURCE_SYSTEMS.items()
    }
    contract = sign_acquisition_contract(
        {
            "schema_version": "lt_input_snapshot.v1",
            "layout": "external_v2",
            "generation_id": "migrated-001",
            "acquisition_id": "migrated-001",
            "available_at_utc": "2026-07-16T08:00:00Z",
            "calibration_eligible": True,
            "source_class": "GOVERNED_ACQUISITION",
            "files": files,
        },
        private_key_path=private,
    )

    with pytest.raises(ValueError, match="requires a governed snapshot schema"):
        validate_lt_input_contract_semantics(contract)


def test_v2_requires_distinct_signed_pit_receipts_and_bound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, contract = _governed_snapshot(
        tmp_path,
        monkeypatch,
        publish_projection=True,
    )

    paths = resolve_lt_input_paths(tmp_path / "project", data_root=data_root)

    assert paths.schema_version == PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA
    assert paths.calibration_eligible is True
    assert paths.available_at_utc == "2026-07-16T08:05:00+00:00"
    assert verify_acquisition_contract(contract)["generation_id"] == "generation-001"


def test_v2_rejects_one_key_for_acquisition_and_trusted_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_key = Ed25519PrivateKey.generate()
    acquisition_private, acquisition_public = _write_keypair(
        tmp_path / "shared-acquisition",
        shared_key,
    )
    timestamp_private, timestamp_public = _write_keypair(
        tmp_path / "shared-time",
        shared_key,
    )
    journal_private, journal_public = _write_keypair(
        tmp_path / "journal",
        Ed25519PrivateKey.generate(),
    )
    monkeypatch.setenv(TRUSTED_ACQUISITION_PUBLIC_KEY_ENV, str(acquisition_public))
    monkeypatch.setenv(TRUSTED_TIME_PUBLIC_KEY_ENV, str(timestamp_public))
    monkeypatch.setenv(TRUSTED_JOURNAL_PUBLIC_KEY_ENV, str(journal_public))
    monkeypatch.setenv(TRUSTED_TIME_JOURNAL_ID_ENV, "journal-001")
    _, unsigned = _unsigned_snapshot(
        tmp_path,
        timestamp_private=timestamp_private,
        journal_private=journal_private,
    )
    signed = sign_acquisition_contract(unsigned, private_key_path=acquisition_private)

    with pytest.raises(
        AcquisitionAuthenticationError,
        match="must use distinct keys",
    ):
        verify_acquisition_contract(signed)


def test_v2_rejects_source_receipt_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract = _governed_snapshot(tmp_path, monkeypatch)
    contract = json.loads(json.dumps(contract))
    contract.pop("acquisition_attestation")
    contract["files"]["epex_ch"]["source_receipt"]["raw_size_bytes"] += 1
    acquisition_private = Path(os.environ["TEST_ACQUISITION_PRIVATE_KEY"])
    resigned = sign_acquisition_contract(contract, private_key_path=acquisition_private)

    with pytest.raises(AcquisitionAuthenticationError, match="time payload hash mismatch"):
        verify_acquisition_contract(resigned)


@pytest.mark.parametrize("role", ["epex_ch", "epex_fr"])
def test_v2_rejects_replayed_raw_derived_path_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    _, contract = _governed_snapshot(tmp_path, monkeypatch)
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    entry = unsigned["files"][role]
    entry["raw_artifact"]["path"] = entry["path"]
    resigned = sign_acquisition_contract(
        unsigned,
        private_key_path=os.environ["TEST_ACQUISITION_PRIVATE_KEY"],
    )

    with pytest.raises(AcquisitionAuthenticationError, match="aliases raw, derived"):
        verify_acquisition_contract(resigned)


@pytest.mark.parametrize("role", ["epex_ch", "epex_fr"])
def test_v2_replay_rejects_fully_resigned_derived_semantic_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    data_root, contract = _governed_snapshot(tmp_path, monkeypatch)
    generation = data_root / "snapshots" / "generation-001"
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    entry = unsigned["files"][role]
    derived_path = generation / entry["path"]
    mutated = pd.read_parquet(derived_path)
    mutated.iloc[0, mutated.columns.get_loc("price_eur_mwh")] += 1.0
    mutated.to_parquet(derived_path, index=True)
    payload = derived_path.read_bytes()
    entry["sha256"] = _sha256(payload)
    entry["size_bytes"] = len(payload)
    config_path = generation / entry["derivation"]["parser_config_path"]
    replay_config = json.loads(config_path.read_text(encoding="utf-8"))
    replay_config["derived_frame_sha256"] = dataframe_sha256(mutated)
    config_path.write_text(
        json.dumps(replay_config, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    entry["derivation"]["parser_config_sha256"] = _sha256(config_path.read_bytes())
    resigned = _resign_snapshot_after_file_change(unsigned)

    with pytest.raises(ValueError, match="quality artifact bindings mismatch"):
        validate_governed_lt_snapshot_bundle(generation, resigned)


def test_v2_replay_rejects_fully_resigned_unapproved_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, contract = _governed_snapshot(tmp_path, monkeypatch)
    generation = data_root / "snapshots" / "generation-001"
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    entry = unsigned["files"]["epex_ch"]
    parser_path = generation / entry["derivation"]["parser_code_path"]
    parser_path.write_bytes(b"def parse(payload):\n    return payload\n")
    entry["derivation"]["parser_code_sha256"] = _sha256(parser_path.read_bytes())
    resigned = _resign_snapshot_after_file_change(unsigned)

    with pytest.raises(ValueError, match="quality artifact bindings mismatch"):
        validate_governed_lt_snapshot_bundle(generation, resigned)


def test_v3_rejects_fully_resigned_unapproved_provider_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, contract = _governed_snapshot(tmp_path, monkeypatch)
    generation = data_root / "snapshots" / "generation-001"
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    entry = unsigned["files"]["epex_ch"]
    parser_path = generation / entry["provider_derivation"]["parser_code_path"]
    parser_path.write_bytes(b"def parse(payload):\n    return payload\n")
    parser_hash = _sha256(parser_path.read_bytes())
    entry["provider_derivation"]["parser_code_sha256"] = parser_hash
    _update_quality_binding(
        generation,
        entry,
        binding="provider_parser_code_sha256",
        value=parser_hash,
    )
    resigned = _resign_snapshot_after_file_change(unsigned)

    with pytest.raises(ValueError, match="provider parser is not the installed allow-listed"):
        validate_governed_lt_snapshot_bundle(generation, resigned)


def test_v3_rejects_fully_resigned_provider_body_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, contract = _governed_snapshot(tmp_path, monkeypatch)
    generation = data_root / "snapshots" / "generation-001"
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    entry = unsigned["files"]["hydro"]
    envelope_path = generation / entry["provider_raw_artifact"]["path"]
    envelope = json.loads(envelope_path.read_bytes())
    document = envelope["documents"][0]
    body = base64.b64decode(document["body_base64"], validate=True)
    mutated_body = body.replace(
        b",4200.0,4000.0,",
        b",4201.0,4000.0,",
        1,
    )
    assert mutated_body != body
    document["body_base64"] = base64.b64encode(mutated_body).decode("ascii")
    document["body_sha256"] = _sha256(mutated_body)
    document["body_size_bytes"] = len(mutated_body)
    envelope_payload = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("ascii")
    envelope_path.write_bytes(envelope_payload)
    envelope_hash = _sha256(envelope_payload)
    entry["provider_raw_artifact"]["sha256"] = envelope_hash
    entry["provider_raw_artifact"]["size_bytes"] = len(envelope_payload)
    receipt = dict(entry["source_receipt"])
    receipt.pop("trusted_time_attestation")
    receipt.pop("receipt_id")
    receipt["raw_sha256"] = envelope_hash
    receipt["raw_size_bytes"] = len(envelope_payload)
    entry["source_receipt"] = sign_source_acquisition_receipt(
        receipt,
        private_key_path=os.environ["TEST_TIMESTAMP_PRIVATE_KEY"],
    )
    _update_quality_binding(
        generation,
        entry,
        binding="provider_raw_sha256",
        value=envelope_hash,
    )
    resigned = _resign_snapshot_after_file_change(unsigned)

    with pytest.raises(ValueError, match="regional totals do not reconcile"):
        validate_governed_lt_snapshot_bundle(generation, resigned)


def test_v3_rejects_fully_resigned_false_quality_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, contract = _governed_snapshot(tmp_path, monkeypatch)
    generation = data_root / "snapshots" / "generation-001"
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    entry = unsigned["files"]["epex_ch"]
    quality = entry["quality_evidence"]
    quality_path = generation / quality["report_path"]
    report = json.loads(quality_path.read_bytes())
    report["metrics"]["bronze"]["row_count"] += 1
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    quality_path.write_bytes(payload)
    quality["report_sha256"] = _sha256(payload)
    resigned = _resign_snapshot_after_file_change(unsigned)

    with pytest.raises(ValueError, match="quality metrics are not exact"):
        validate_governed_lt_snapshot_bundle(generation, resigned)


def test_v3_rejects_fully_resigned_epex_cadence_metadata_stripping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, contract = _governed_snapshot(tmp_path, monkeypatch)
    generation = data_root / "snapshots" / "generation-001"
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    entry = unsigned["files"]["epex_ch"]

    raw_path = generation / entry["raw_artifact"]["path"]
    raw = pd.read_parquet(raw_path)
    raw.attrs.clear()
    raw.to_parquet(raw_path, index=True)
    raw_payload = raw_path.read_bytes()
    entry["raw_artifact"]["sha256"] = _sha256(raw_payload)
    entry["raw_artifact"]["size_bytes"] = len(raw_payload)

    derived_path = generation / entry["path"]
    derived = pd.read_parquet(derived_path)
    derived.attrs.clear()
    derived.to_parquet(derived_path, index=True)
    derived_payload = derived_path.read_bytes()
    entry["sha256"] = _sha256(derived_payload)
    entry["size_bytes"] = len(derived_payload)

    replay_config_path = generation / entry["derivation"]["parser_config_path"]
    replay_config_payload = json.dumps(
        build_replay_config(
            role="epex_ch",
            source_system=ENERGY_CHARTS_SOURCE_SYSTEM,
            raw_frame=raw,
            derived_frame=derived,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    replay_config_path.write_bytes(replay_config_payload)
    entry["derivation"]["parser_config_sha256"] = _sha256(replay_config_payload)

    quality = entry["quality_evidence"]
    quality_path = generation / quality["report_path"]
    report = json.loads(quality_path.read_bytes())
    report["bindings"]["bronze_sha256"] = _sha256(raw_payload)
    report["bindings"]["feature_parser_config_sha256"] = _sha256(
        replay_config_payload
    )
    report["bindings"]["derived_sha256"] = _sha256(derived_payload)
    report["metrics"] = {
        "bronze": _frame_quality_metrics(raw),
        "derived": _frame_quality_metrics(derived),
    }
    quality_payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    quality_path.write_bytes(quality_payload)
    quality["report_sha256"] = _sha256(quality_payload)
    resigned = _resign_snapshot_after_file_change(unsigned)

    with pytest.raises(
        ValueError,
        match="resolution provenance is missing or ambiguous",
    ):
        validate_governed_lt_snapshot_bundle(generation, resigned)


def test_v2_rejects_checkpoint_issued_after_snapshot_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract = _governed_snapshot(tmp_path, monkeypatch)
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    unsigned["files"]["hydro"]["available_at_utc"] = "2026-07-16T08:04:00Z"
    unsigned["files"]["hydro"]["derivation"]["derived_at_utc"] = "2026-07-16T08:04:00Z"
    unsigned["files"]["hydro"]["provider_derivation"]["derived_at_utc"] = "2026-07-16T08:04:00Z"
    unsigned["available_at_utc"] = "2026-07-16T08:04:00Z"
    resigned = _resign_snapshot_after_file_change(unsigned)

    with pytest.raises(
        AcquisitionAuthenticationError,
        match="maximum governed availability",
    ):
        verify_acquisition_contract(resigned)


def test_v2_rejects_signed_but_disconnected_receipt_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, contract = _governed_snapshot(tmp_path, monkeypatch)
    unsigned = json.loads(json.dumps(contract))
    unsigned.pop("acquisition_attestation")
    ordered_entries = sorted(
        unsigned["files"].values(),
        key=lambda entry: entry["source_receipt"]["sequence"],
    )
    broken = dict(ordered_entries[1]["source_receipt"])
    broken.pop("trusted_time_attestation")
    broken.pop("receipt_id")
    broken["previous_receipt_id"] = "f" * 64
    ordered_entries[1]["source_receipt"] = sign_source_acquisition_receipt(
        broken,
        private_key_path=os.environ["TEST_TIMESTAMP_PRIVATE_KEY"],
    )
    receipt_ids = [str(entry["source_receipt"]["receipt_id"]) for entry in ordered_entries]
    unsigned["source_journal_checkpoint"] = sign_source_journal_checkpoint(
        {
            "schema_version": "source_journal_checkpoint.v1",
            "journal_id": "journal-001",
            "acquisition_id": "acquisition-001",
            "first_sequence": 1,
            "last_sequence": len(receipt_ids),
            "previous_receipt_id": None,
            "receipt_ids": receipt_ids,
            "head_receipt_id": receipt_ids[-1],
            "issued_at_utc": "2026-07-16T08:05:00Z",
            "bundle_root_sha256": governed_snapshot_bundle_root_sha256(unsigned["files"]),
        },
        private_key_path=os.environ["TEST_JOURNAL_PRIVATE_KEY"],
    )
    resigned = sign_acquisition_contract(
        unsigned,
        private_key_path=os.environ["TEST_ACQUISITION_PRIVATE_KEY"],
    )

    with pytest.raises(AcquisitionAuthenticationError, match="chain is not linked"):
        verify_acquisition_contract(resigned)


def test_v2_rejects_tampered_quality_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, _ = _governed_snapshot(
        tmp_path,
        monkeypatch,
        publish_projection=True,
    )
    report = data_root / "snapshots" / "generation-001" / "quality" / "epex_ch.json"
    report.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"hash mismatch: epex_ch\.quality_report"):
        resolve_lt_input_paths(tmp_path / "project", data_root=data_root)


def test_publisher_copies_signed_bundle_without_private_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    before = _tree_hashes(source)
    target = tmp_path / "target"
    target.mkdir()
    for name in (
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH",
        "TEST_ACQUISITION_PRIVATE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    result = publish_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )

    assert result["generation_id"] == "generation-001"
    assert result["resumed_existing_generation"] is False
    assert _tree_hashes(source) == before
    with pytest.raises(ValueError, match="external CAS v2 pointer"):
        resolve_lt_input_paths(tmp_path / "project", data_root=target)
    event = target / "views" / "pfc_lt" / "events" / f"{result['publication_event_id']}.json"
    assert event.is_file()


def test_publisher_rejects_stale_pointer_expectation_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    target.mkdir()
    publish_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )

    with pytest.raises(SnapshotPublicationConflict, match="caller expectation"):
        publish_governed_lt_snapshot(
            source_bundle=source,
            data_root=target,
            expected_current_event_id=None,
            operation_id=_operation_id(2),
        )

    assert not list((target / "snapshots").glob(".*.staging-*"))


def test_publisher_is_idempotent_when_generation_is_already_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    target.mkdir()
    first = publish_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )

    second = publish_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_current_event_id=str(first["publication_event_id"]),
        operation_id=_operation_id(2),
    )

    assert second["already_current"] is True
    assert second["pointer_sha256"] == first["pointer_sha256"]
    assert second["publication_event_id"] == first["publication_event_id"]
    assert len(list((target / "views" / "pfc_lt" / "events").glob("*.json"))) == 1


def test_publisher_exact_retry_survives_lost_success_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    target.mkdir()
    first = publish_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )

    retry = publish_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )

    assert retry["exact_retry"] is True
    assert retry["commit_status"] == "COMMITTED"
    assert retry["publication_event_id"] == first["publication_event_id"]
    assert retry["pointer_sha256"] == first["pointer_sha256"]


def test_publisher_rejects_operation_id_reuse_with_different_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source_a = source_root / "snapshots" / "generation-001"
    source_b = _second_generation(source_a, tmp_path / "source-b")
    target = tmp_path / "target"
    target.mkdir()
    first = publish_governed_lt_snapshot(
        source_bundle=source_a,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )

    with pytest.raises(SnapshotPublicationConflict, match="different inputs"):
        publish_governed_lt_snapshot(
            source_bundle=source_b,
            data_root=target,
            expected_current_event_id=str(first["publication_event_id"]),
            operation_id=_operation_id(1),
        )


def test_publisher_revision_prevents_pointer_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source_a = source_root / "snapshots" / "generation-001"
    source_b = tmp_path / "source-b"
    shutil.copytree(source_a, source_b)
    contract_b = json.loads((source_b / "lt_input_snapshot.json").read_text(encoding="utf-8"))
    contract_b.pop("acquisition_attestation")
    contract_b["generation_id"] = "generation-002"
    _write_json(
        source_b / "lt_input_snapshot.json",
        sign_acquisition_contract(
            contract_b,
            private_key_path=os.environ["TEST_ACQUISITION_PRIVATE_KEY"],
        ),
    )
    target = tmp_path / "target"
    target.mkdir()
    first_a = publish_governed_lt_snapshot(
        source_bundle=source_a,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )
    second_b = publish_governed_lt_snapshot(
        source_bundle=source_b,
        data_root=target,
        expected_current_event_id=str(first_a["publication_event_id"]),
        operation_id=_operation_id(2),
    )

    with pytest.raises(SnapshotPublicationConflict, match="caller expectation"):
        publish_governed_lt_snapshot(
            source_bundle=source_a,
            data_root=target,
            expected_current_event_id=str(first_a["publication_event_id"]),
            operation_id=_operation_id(3),
        )

    third_a = publish_governed_lt_snapshot(
        source_bundle=source_a,
        data_root=target,
        expected_current_event_id=str(second_b["publication_event_id"]),
        operation_id=_operation_id(3),
    )
    assert third_a["pointer_sha256"] != first_a["pointer_sha256"]
    pointer = json.loads((target / "views" / "pfc_lt" / "current.json").read_text())
    assert pointer["revision"] == 3


def test_publisher_rejects_unbound_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    (source / "unbound.txt").write_text("not governed", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(ValueError, match="file set is not fully bound"):
        publish_governed_lt_snapshot(
            source_bundle=source,
            data_root=target,
            expected_current_event_id=None,
            operation_id=_operation_id(1),
        )


def test_publisher_fails_closed_when_lock_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    lock = target / "views" / "pfc_lt" / ".publish.lock"
    lock.mkdir(parents=True)

    with pytest.raises(SnapshotPublicationBusy, match="lock is busy"):
        publish_governed_lt_snapshot(
            source_bundle=source,
            data_root=target,
            expected_current_event_id=None,
            operation_id=_operation_id(1),
        )

    assert not (target / "snapshots" / "generation-001").exists()
    assert not list((target / "snapshots").glob(".*.staging-*"))


def test_publisher_never_writes_through_hardlinked_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    view = target / "views" / "pfc_lt"
    view.mkdir(parents=True)
    victim = tmp_path / "external-victim.bin"
    victim.write_bytes(b"")
    os.link(victim, view / ".publish.lock")

    with pytest.raises(SnapshotPublicationBusy, match="lock is busy"):
        publish_governed_lt_snapshot(
            source_bundle=source,
            data_root=target,
            expected_current_event_id=None,
            operation_id=_operation_id(1),
        )

    assert victim.read_bytes() == b""
    assert victim.stat().st_nlink == 2


def test_resolver_rejects_unsigned_forged_publication_genesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, _ = _governed_snapshot(
        tmp_path,
        monkeypatch,
        publish_projection=True,
    )
    pointer = json.loads(
        (data_root / "views" / "pfc_lt" / "current.json").read_text(encoding="utf-8")
    )
    intent_path = (
        data_root / "views" / "pfc_lt" / "intents" / f"{pointer['publication_intent_id']}.json"
    )
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent.pop("signature")
    intent_path.write_bytes(canonical_json(intent))

    with pytest.raises(SnapshotPublicationStateError, match="authentication failed"):
        resolve_lt_input_paths(tmp_path / "project", data_root=data_root)


def test_resolver_rejects_pointer_rollback_behind_external_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source_a = source_root / "snapshots" / "generation-001"
    source_b = _second_generation(source_a, tmp_path / "source-b")
    target = tmp_path / "target"
    target.mkdir()
    first = publish_governed_lt_snapshot(
        source_bundle=source_a,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )
    first_pointer = (target / "views" / "pfc_lt" / "current.json").read_bytes()
    publish_governed_lt_snapshot(
        source_bundle=source_b,
        data_root=target,
        expected_current_event_id=str(first["publication_event_id"]),
        operation_id=_operation_id(2),
    )
    (target / "views" / "pfc_lt" / "current.json").write_bytes(first_pointer)

    with pytest.raises(ValueError, match="external CAS v2 pointer"):
        resolve_lt_input_paths(tmp_path / "project", data_root=target)


def test_resolver_rejects_truncated_external_anchor_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source_a = source_root / "snapshots" / "generation-001"
    source_b = _second_generation(source_a, tmp_path / "source-b")
    target = tmp_path / "target"
    target.mkdir()
    first = publish_governed_lt_snapshot(
        source_bundle=source_a,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )
    publish_governed_lt_snapshot(
        source_bundle=source_b,
        data_root=target,
        expected_current_event_id=str(first["publication_event_id"]),
        operation_id=_operation_id(2),
    )
    anchors = sorted(publication_anchor_directory(target, create=False).glob("*.json"))
    anchors[-1].unlink()

    with pytest.raises(ValueError, match="external CAS v2 pointer"):
        resolve_lt_input_paths(tmp_path / "project", data_root=target)


def test_publication_anchor_rejects_signed_receipt_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, _ = _governed_snapshot(
        tmp_path,
        monkeypatch,
        publish_projection=True,
    )
    pointer = json.loads(
        (data_root / "views" / "pfc_lt" / "current.json").read_text(encoding="utf-8")
    )
    receipt_path = (
        data_root / "views" / "pfc_lt" / "receipts" / f"{pointer['anchor_receipt_id']}.json"
    )
    fork = json.loads(receipt_path.read_text(encoding="utf-8"))
    fork.pop("signature")
    fork.pop("receipt_id")
    fork["generation_id"] = "generation-fork"
    signed_fork = sign_snapshot_anchor_receipt(
        fork,
        private_key_path=os.environ["TEST_PUBLICATION_ANCHOR_PRIVATE_KEY"],
    )
    receipt_path.write_bytes(canonical_json(signed_fork))

    with pytest.raises(SnapshotPublicationStateError, match="exact publication intent"):
        resolve_lt_input_paths(tmp_path / "project", data_root=data_root)


def test_publisher_requires_distinct_publication_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    publication_public = os.environ[PUBLICATION_TRUSTED_PUBLIC_KEY_ENV]
    monkeypatch.setenv(TRUSTED_ACQUISITION_PUBLIC_KEY_ENV, publication_public)
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(ValueError, match="overlaps another governed authority"):
        publish_governed_lt_snapshot(
            source_bundle=source_root / "snapshots" / "generation-001",
            data_root=target,
            expected_current_event_id=None,
            operation_id=_operation_id(1),
        )


def test_publisher_rejects_exposed_acquisition_private_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        os.environ["TEST_ACQUISITION_PRIVATE_KEY"],
    )
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(ValueError, match="forbidden private keys"):
        publish_governed_lt_snapshot(
            source_bundle=source_root / "snapshots" / "generation-001",
            data_root=target,
            expected_current_event_id=None,
            operation_id=_operation_id(1),
        )


def test_external_publisher_runtime_rejects_request_key_outside_prepare_and_bootstrap_key_always(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _governed_snapshot(tmp_path / "source", monkeypatch)
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)

    with pytest.raises(ValueError, match="forbidden private keys"):
        snapshot_publisher.assert_snapshot_publisher_runtime_identity(phase="cas")

    monkeypatch.delenv(PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.setenv(
        "PFC_DATA_PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_PATH",
        os.environ["TEST_ACQUISITION_PRIVATE_KEY"],
    )
    with pytest.raises(ValueError, match="forbidden private keys"):
        snapshot_publisher.assert_snapshot_publisher_runtime_identity(phase="finalize")


def test_external_publisher_registry_rejects_historical_cross_role_key_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _governed_snapshot(tmp_path / "source", monkeypatch)
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.delenv(PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV)
    acquisition_keyring = tmp_path / "acquisition-keyring"
    acquisition_keyring.mkdir()
    (acquisition_keyring / "historical-anchor.pem").write_bytes(
        Path(os.environ[PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV]).read_bytes()
    )
    monkeypatch.setenv(
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_DIR",
        str(acquisition_keyring),
    )

    with pytest.raises(ValueError, match="overlapping authorities"):
        snapshot_publisher.assert_snapshot_publisher_runtime_identity(phase="finalize")


def test_external_publisher_registry_rejects_missing_governed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _governed_snapshot(tmp_path / "source", monkeypatch)
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.delenv(PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.delenv("PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH")

    with pytest.raises(ValueError, match="trust registry is incomplete"):
        snapshot_publisher.assert_snapshot_publisher_runtime_identity(phase="finalize")


def test_external_publisher_freezes_trust_files_before_downstream_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _governed_snapshot(tmp_path / "source", monkeypatch)
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.delenv(PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV)
    original_anchor = Path(os.environ[PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV])
    original_payload = original_anchor.read_bytes()
    anchor_private = Path(os.environ["TEST_PUBLICATION_ANCHOR_PRIVATE_KEY"])
    model_governance = Path(os.environ["PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH"])

    snapshot_publisher.assert_snapshot_publisher_runtime_identity(phase="finalize")
    captured_anchor = Path(os.environ[PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV])
    original_anchor.write_bytes(model_governance.read_bytes())
    captured_anchor.write_bytes(model_governance.read_bytes())

    assert captured_anchor != original_anchor
    assert captured_anchor.read_bytes() != original_payload
    assert publication_public_key_id(captured_anchor) == publication_private_key_id(anchor_private)
    assert publication_public_key_id(original_anchor) != publication_public_key_id(captured_anchor)


@pytest.mark.parametrize(
    "tier2_environment",
    [
        "PFC_TIER2_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_TIER2_BASE_MODEL_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_TIER2_SELECTION_INPUT_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_TIER2_TIMESTAMP_JOURNAL_WITNESS_TRUSTED_PUBLIC_KEY_PATH",
    ],
)
def test_external_publisher_registry_rejects_each_tier2_cross_role_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier2_environment: str,
) -> None:
    _governed_snapshot(tmp_path / "source", monkeypatch)
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.delenv(PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.setenv(
        tier2_environment,
        os.environ[PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV],
    )

    with pytest.raises(ValueError, match="overlapping authorities"):
        snapshot_publisher.assert_snapshot_publisher_runtime_identity(phase="finalize")


def test_publisher_repairs_projection_only_for_exact_committed_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    target.mkdir()
    original_write_atomic = snapshot_publisher._write_atomic

    def fail_pointer_projection(path: Path, payload: bytes) -> None:
        if path.name == "current.json":
            raise RuntimeError("injected failure after external anchor commit")
        original_write_atomic(path, payload)

    with monkeypatch.context() as injected:
        injected.setattr(snapshot_publisher, "_write_atomic", fail_pointer_projection)
        with pytest.raises(RuntimeError, match="after external anchor commit"):
            publish_governed_lt_snapshot(
                source_bundle=source,
                data_root=target,
                expected_current_event_id=None,
                operation_id=_operation_id(1),
            )

    assert not (target / "views" / "pfc_lt" / "current.json").exists()
    assert len(list(publication_anchor_directory(target, create=False).glob("*.json"))) == 1
    with pytest.raises(SnapshotPublicationConflict, match="different projection repair"):
        publish_governed_lt_snapshot(
            source_bundle=source,
            data_root=target,
            expected_current_event_id=None,
            operation_id=_operation_id(2),
        )

    repaired = publish_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_current_event_id=None,
        operation_id=_operation_id(1),
    )
    assert repaired["projection_repaired"] is True
    assert repaired["commit_status"] == "COMMITTED"
    with pytest.raises(ValueError, match="external CAS v2 pointer"):
        resolve_lt_input_paths(tmp_path / "project", data_root=target)


def test_external_cas_prepare_finalize_and_exact_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    source = source_root / "snapshots" / "generation-001"
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
    predecessor_payload = _write_external_predecessor(target)

    prepared = prepare_governed_lt_snapshot(
        source_bundle=source,
        data_root=target,
        expected_anchor_sequence=1,
        expected_anchor_receipt_id="0" * 64,
        operation_id=_operation_id(11),
        operation_created_at_utc="2026-07-16T09:00:00+00:00",
    )

    assert prepared["status"] == "PREPARED_NOT_COMMITTED"
    assert prepared["schema_version"] == "lt_snapshot_publication_cli_result.v1"
    assert prepared["command"] == "prepare"
    assert prepared["commit_status"] == "NOT_COMMITTED"
    assert (target / "views" / "pfc_lt" / "current.json").read_bytes() == predecessor_payload
    receipt_path, observation_path = _external_cas_evidence(
        tmp_path,
        intent_path=Path(str(prepared["intent_path"])),
    )
    monkeypatch.setattr(
        publication_state_module,
        "_reference_utc",
        lambda _: datetime.fromisoformat("2026-07-16T09:03:30+00:00"),
    )

    finalized = finalize_governed_lt_snapshot(
        data_root=target,
        intent_path=Path(str(prepared["intent_path"])),
        anchor_receipt_path=receipt_path,
        head_observation_path=observation_path,
        expected_head_challenge_nonce="4" * 64,
    )
    retried = finalize_governed_lt_snapshot(
        data_root=target,
        intent_path=Path(str(prepared["intent_path"])),
        anchor_receipt_path=receipt_path,
        head_observation_path=observation_path,
        expected_head_challenge_nonce="4" * 64,
    )

    pointer = json.loads((target / "views" / "pfc_lt" / "current.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="fresh external anchor HEAD observation"):
        resolve_lt_input_paths(tmp_path / "project", data_root=target)
    resolved = resolve_lt_input_paths(
        tmp_path / "project",
        data_root=target,
        publication_head_observation=observation_path,
        publication_head_challenge_nonce="4" * 64,
    )
    assert finalized["status"] == "FINALIZED"
    assert retried["status"] == "ALREADY_FINALIZED"
    assert finalized["schema_version"] == "lt_snapshot_publication_cli_result.v1"
    assert finalized["command"] == "finalize"
    assert finalized["commit_status"] == "COMMITTED"
    assert retried["projection_status"] == "ALREADY_FINALIZED"
    assert pointer["schema_version"] == "lt_data_pointer.v2"
    assert pointer["operation_id"] == _operation_id(11)
    assert finalized["production_authorization"] is False
    assert resolved.publication_intent_receipt is not None
    assert resolved.publication_anchor_receipt is not None
    assert resolved.publication_head_observation_receipt is not None


def test_legacy_bootstrap_is_exact_authorized_and_remains_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root, governed_contract = _governed_snapshot(tmp_path, monkeypatch)
    generation_id = "generation-001"
    generation = data_root / "snapshots" / generation_id
    migrated_files = {}
    for role, raw_entry in governed_contract["files"].items():
        entry = dict(raw_entry)
        migrated_files[role] = {
            "acquisition_id": "migrated-seed",
            "available_at_utc": "2026-07-16T08:58:00+00:00",
            "calibration_eligible": False,
            "path": entry["path"],
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }
    migrated_contract = {
        "schema_version": "lt_input_snapshot.v1",
        "generation_id": generation_id,
        "acquisition_id": "migrated-seed",
        "created_at_utc": "2026-07-16T08:58:00+00:00",
        "available_at_utc": "2026-07-16T08:58:00+00:00",
        "layout": "external_v2",
        "source_class": BOOTSTRAP_REQUIRED_SOURCE_CLASS,
        "provenance": "MIGRATED_LEGACY_CACHE_UNVERIFIED_NOT_CALIBRATION_ELIGIBLE",
        "calibration_eligible": False,
        "files": migrated_files,
    }
    contract_path = generation / "lt_input_snapshot.json"
    contract_payload = canonical_json(migrated_contract)
    contract_path.write_bytes(contract_payload)
    pointer_payload = canonical_json(
        {
            "schema_version": "lt_data_pointer.v1",
            "generation_id": generation_id,
            "contract_path": f"snapshots/{generation_id}/lt_input_snapshot.json",
            "contract_sha256": _sha256(contract_payload),
        }
    )
    pointer_path = data_root / "views" / "pfc_lt" / "current.json"
    _write_bytes(pointer_path, pointer_payload)
    bound = publisher_module._discover_bound_files(generation)
    _, inventory_payload = publisher_module._generation_inventory(
        generation,
        generation_id=generation_id,
        bound=bound,
    )
    operation_id = _operation_id(13)
    bootstrap_private, bootstrap_public = _write_keypair(
        tmp_path / "bootstrap-authority",
        Ed25519PrivateKey.generate(),
    )
    monkeypatch.setenv(
        PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_ENV,
        str(bootstrap_public),
    )
    authorization = sign_snapshot_bootstrap_authorization(
        {
            "schema_version": PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
            "publication_domain_sha256": publication_domain_sha256(),
            "transition_type": BOOTSTRAP_TRANSITION_TYPE,
            "use_policy": BOOTSTRAP_USE_POLICY,
            "operation_id": operation_id,
            "expected_anchor_sequence": 0,
            "expected_anchor_receipt_id": None,
            "legacy_pointer_sha256": _sha256(pointer_payload),
            "generation_id": generation_id,
            "contract_path": f"snapshots/{generation_id}/lt_input_snapshot.json",
            "contract_sha256": _sha256(contract_payload),
            "generation_inventory_sha256": _sha256(inventory_payload),
            "required_source_class": BOOTSTRAP_REQUIRED_SOURCE_CLASS,
            "required_calibration_eligible": False,
            "issued_at_utc": "2026-07-16T08:59:00+00:00",
            "expires_at_utc": "2026-07-16T09:05:00+00:00",
        },
        private_key_path=bootstrap_private,
    )
    authorization_path = tmp_path / "bootstrap-authorization.json"
    authorization_path.write_bytes(canonical_json(authorization))
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
    monkeypatch.setattr(
        publisher_module,
        "_reference_utc",
        lambda: datetime.fromisoformat("2026-07-16T09:02:00+00:00"),
    )

    prepared = prepare_governed_lt_snapshot(
        source_bundle=None,
        data_root=data_root,
        expected_anchor_sequence=0,
        expected_anchor_receipt_id=None,
        operation_id=operation_id,
        operation_created_at_utc="2026-07-16T09:00:00+00:00",
        expected_legacy_pointer_sha256=_sha256(pointer_payload),
        bootstrap_authorization_path=authorization_path,
    )
    assert pointer_path.read_bytes() == pointer_payload
    assert prepared["bootstrap_authorization_id"] == authorization["authorization_id"]
    receipt_path, observation_path = _external_cas_evidence(
        tmp_path,
        intent_path=Path(str(prepared["intent_path"])),
    )
    monkeypatch.setattr(
        publication_state_module,
        "_reference_utc",
        lambda _: datetime.fromisoformat("2026-07-16T09:03:00+00:00"),
    )

    finalized = finalize_governed_lt_snapshot(
        data_root=data_root,
        intent_path=Path(str(prepared["intent_path"])),
        anchor_receipt_path=receipt_path,
        head_observation_path=observation_path,
        expected_head_challenge_nonce="4" * 64,
    )

    projected = json.loads(pointer_path.read_bytes())
    persisted_contract = json.loads(contract_path.read_bytes())
    assert finalized["status"] == "FINALIZED"
    assert projected["anchor_sequence"] == 1
    assert persisted_contract["source_class"] == BOOTSTRAP_REQUIRED_SOURCE_CLASS
    assert persisted_contract["calibration_eligible"] is False
    assert (
        data_root
        / "views"
        / "pfc_lt"
        / "bootstrap-authorizations"
        / f"{authorization['authorization_id']}.json"
    ).read_bytes() == canonical_json(authorization)


@pytest.mark.parametrize(
    "failure_stage",
    ["inventory", "generation", "evidence_archive", "pointer_projection"],
)
def test_external_cas_finalize_reports_committed_repair_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    source_root, _ = _governed_snapshot(tmp_path / "source", monkeypatch)
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
    _write_external_predecessor(target)
    prepared = prepare_governed_lt_snapshot(
        source_bundle=source_root / "snapshots" / "generation-001",
        data_root=target,
        expected_anchor_sequence=1,
        expected_anchor_receipt_id="0" * 64,
        operation_id=_operation_id(12),
        operation_created_at_utc="2026-07-16T09:00:00+00:00",
    )
    receipt_path, observation_path = _external_cas_evidence(
        tmp_path,
        intent_path=Path(str(prepared["intent_path"])),
    )
    monkeypatch.setattr(
        publication_state_module,
        "_reference_utc",
        lambda _: datetime.fromisoformat("2026-07-16T09:03:30+00:00"),
    )
    if failure_stage == "inventory":
        original_read = publisher_module._read_stable_regular_file

        def fail_inventory(path: Path, *, label: str) -> bytes:
            if label == "generation inventory":
                raise OSError("inventory unavailable")
            return original_read(path, label=label)

        monkeypatch.setattr(publisher_module, "_read_stable_regular_file", fail_inventory)
    elif failure_stage == "generation":
        monkeypatch.setattr(
            publisher_module,
            "_verify_existing_generation",
            lambda *_: (_ for _ in ()).throw(OSError("generation unavailable")),
        )
    elif failure_stage == "evidence_archive":
        monkeypatch.setattr(
            publisher_module,
            "_write_immutable_or_identical",
            lambda *_: (_ for _ in ()).throw(OSError("evidence archive unavailable")),
        )
    else:
        monkeypatch.setattr(
            publisher_module,
            "_write_atomic",
            lambda *_: (_ for _ in ()).throw(OSError("projection unavailable")),
        )

    with pytest.raises(SnapshotPublicationRepairRequired) as captured:
        finalize_governed_lt_snapshot(
            data_root=target,
            intent_path=Path(str(prepared["intent_path"])),
            anchor_receipt_path=receipt_path,
            head_observation_path=observation_path,
            expected_head_challenge_nonce="4" * 64,
        )

    result = captured.value.result
    assert result == {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": "COMMITTED_REPAIR_REQUIRED",
        "command": "finalize",
        "commit_status": "COMMITTED",
        "projection_status": "LOCAL_PROJECTION_REPAIR_REQUIRED",
        "retry_disposition": "RETRY_FINALIZE_WITH_FRESH_HEAD",
        "operation_id": _operation_id(12),
        "intent_id": json.loads(Path(str(prepared["intent_path"])).read_bytes())["intent_id"],
        "anchor_receipt_id": json.loads(receipt_path.read_bytes())["receipt_id"],
        "anchor_sequence": 2,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path.read_bytes()),
        "error": {
            "type": "OSError",
            "message": {
                "inventory": "inventory unavailable",
                "generation": "generation unavailable",
                "evidence_archive": "evidence archive unavailable",
                "pointer_projection": "projection unavailable",
            }[failure_stage],
        },
        "local_durability": durable_artifact_durability_classification(),
        "production_authorization": False,
    }


def _external_cas_evidence(
    tmp_path: Path,
    *,
    intent_path: Path,
) -> tuple[Path, Path]:
    intent_payload = intent_path.read_bytes()
    intent = json.loads(intent_payload)
    anchor_private = Path(os.environ["TEST_PUBLICATION_ANCHOR_PRIVATE_KEY"])
    receipt = sign_snapshot_anchor_receipt(
        {
            "schema_version": "lt_snapshot_anchor_receipt.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "sequence": int(intent["expected_anchor_sequence"]) + 1,
            "previous_receipt_id": intent["expected_anchor_receipt_id"],
            "operation_id": intent["operation_id"],
            "intent_id": intent["intent_id"],
            "intent_sha256": _sha256(intent_payload),
            "transition_type": intent["transition_type"],
            "generation_id": intent["generation_id"],
            "contract_sha256": intent["contract_sha256"],
            "generation_inventory_sha256": intent["generation_inventory_sha256"],
            "legacy_pointer_sha256": intent["legacy_pointer_sha256"],
            "bootstrap_authorization_id": intent["bootstrap_authorization_id"],
            "bootstrap_authorization_sha256": intent["bootstrap_authorization_sha256"],
            "committed_at_utc": "2026-07-16T09:00:01+00:00",
        },
        private_key_path=anchor_private,
    )
    receipt_payload = canonical_json(receipt)
    pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=_sha256(receipt_payload),
    )
    observation = sign_snapshot_anchor_head_observation(
        {
            "schema_version": "lt_snapshot_anchor_head_observation.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "receipt_id": receipt["receipt_id"],
            "receipt_sha256": _sha256(receipt_payload),
            "sequence": receipt["sequence"],
            "challenge_nonce": "4" * 64,
            "challenge_sha256": external_head_challenge_sha256(
                pointer,
                challenge_nonce="4" * 64,
            ),
            "observed_at_utc": "2026-07-16T09:00:02+00:00",
            "expires_at_utc": "2026-07-16T09:05:02+00:00",
        },
        private_key_path=anchor_private,
    )
    receipt_path = tmp_path / "external-anchor" / "receipt.json"
    observation_path = tmp_path / "external-anchor" / "observation.json"
    _write_bytes(receipt_path, receipt_payload)
    _write_bytes(observation_path, canonical_json(observation))
    return receipt_path, observation_path


def _write_external_predecessor(data_root: Path) -> bytes:
    payload = canonical_json(
        {
            "schema_version": "lt_data_pointer.v2",
            "anchor_receipt_id": "0" * 64,
            "anchor_receipt_sha256": "1" * 64,
            "anchor_sequence": 1,
            "publication_intent_id": "2" * 64,
            "operation_id": _operation_id(90),
            "generation_id": "legacy-bootstrap",
            "contract_path": "snapshots/legacy-bootstrap/lt_input_snapshot.json",
            "contract_sha256": "3" * 64,
            "generation_inventory_sha256": "4" * 64,
        }
    )
    _write_bytes(data_root / "views" / "pfc_lt" / "current.json", payload)
    return payload


def _governed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    publish_projection: bool = False,
) -> tuple[Path, dict[str, object]]:
    acquisition_private, acquisition_public = _write_keypair(
        tmp_path / "acquisition",
        Ed25519PrivateKey.generate(),
    )
    timestamp_private, timestamp_public = _write_keypair(
        tmp_path / "timestamp",
        Ed25519PrivateKey.generate(),
    )
    journal_private, journal_public = _write_keypair(
        tmp_path / "journal",
        Ed25519PrivateKey.generate(),
    )
    publication_private, publication_public = _write_keypair(
        tmp_path / "publication",
        Ed25519PrivateKey.generate(),
    )
    request_private, request_public = _write_keypair(
        tmp_path / "publication-request",
        Ed25519PrivateKey.generate(),
    )
    anchor_private, anchor_public = _write_keypair(
        tmp_path / "publication-anchor",
        Ed25519PrivateKey.generate(),
    )
    _, bootstrap_public = _write_keypair(
        tmp_path / "publication-bootstrap",
        Ed25519PrivateKey.generate(),
    )
    for index, environment_name in enumerate(
        (
            "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_BASE_MODEL_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_SELECTION_INPUT_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
            "PFC_TIER2_TIMESTAMP_JOURNAL_WITNESS_TRUSTED_PUBLIC_KEY_PATH",
        )
    ):
        _, governed_public = _write_keypair(
            tmp_path / f"governed-authority-{index}",
            Ed25519PrivateKey.generate(),
        )
        monkeypatch.setenv(environment_name, str(governed_public))
    publication_anchor_root = tmp_path / "publication-anchors"
    publication_anchor_root.mkdir()
    monkeypatch.setenv(TRUSTED_ACQUISITION_PUBLIC_KEY_ENV, str(acquisition_public))
    monkeypatch.setenv(TRUSTED_TIME_PUBLIC_KEY_ENV, str(timestamp_public))
    monkeypatch.setenv(TRUSTED_JOURNAL_PUBLIC_KEY_ENV, str(journal_public))
    monkeypatch.setenv(TRUSTED_TIME_JOURNAL_ID_ENV, "journal-001")
    monkeypatch.setenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV, str(publication_private))
    monkeypatch.setenv(PUBLICATION_TRUSTED_PUBLIC_KEY_ENV, str(publication_public))
    monkeypatch.setenv(PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV, str(request_private))
    monkeypatch.setenv(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV, str(request_public))
    monkeypatch.setenv(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV, str(anchor_public))
    monkeypatch.setenv(
        PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_ENV,
        str(bootstrap_public),
    )
    monkeypatch.setenv(PUBLICATION_ANCHOR_ROOT_ENV, str(publication_anchor_root))
    monkeypatch.setenv(
        PUBLICATION_DOMAIN_ID_ENV,
        "8ed047c1-a150-4e88-ae79-c2cf567108cc",
    )
    monkeypatch.setenv("TEST_ACQUISITION_PRIVATE_KEY", str(acquisition_private))
    monkeypatch.setenv("TEST_TIMESTAMP_PRIVATE_KEY", str(timestamp_private))
    monkeypatch.setenv("TEST_JOURNAL_PRIVATE_KEY", str(journal_private))
    monkeypatch.setenv("TEST_PUBLICATION_ANCHOR_PRIVATE_KEY", str(anchor_private))
    data_root, unsigned = _unsigned_snapshot(
        tmp_path,
        timestamp_private=timestamp_private,
        journal_private=journal_private,
    )
    contract = sign_acquisition_contract(unsigned, private_key_path=acquisition_private)
    contract_path = data_root / "snapshots" / "generation-001" / "lt_input_snapshot.json"
    _write_json(contract_path, contract)
    if publish_projection:
        monkeypatch.delenv(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)
        _write_external_predecessor(data_root)
        prepared = prepare_governed_lt_snapshot(
            source_bundle=contract_path.parent,
            data_root=data_root,
            expected_anchor_sequence=1,
            expected_anchor_receipt_id="0" * 64,
            operation_id=_operation_id(91),
            operation_created_at_utc="2026-07-16T09:00:00+00:00",
        )
        receipt_path, observation_path = _external_cas_evidence(
            tmp_path,
            intent_path=Path(str(prepared["intent_path"])),
        )
        monkeypatch.setattr(
            publication_state_module,
            "_reference_utc",
            lambda _: datetime.fromisoformat("2026-07-16T09:03:00+00:00"),
        )
        finalize_governed_lt_snapshot(
            data_root=data_root,
            intent_path=Path(str(prepared["intent_path"])),
            anchor_receipt_path=receipt_path,
            head_observation_path=observation_path,
            expected_head_challenge_nonce="4" * 64,
        )
        monkeypatch.setenv(
            PUBLICATION_HEAD_OBSERVATION_PATH_ENV,
            str(observation_path),
        )
        monkeypatch.setenv(PUBLICATION_HEAD_CHALLENGE_NONCE_ENV, "4" * 64)
    return data_root, contract


def _write_signed_publication_projection(
    data_root: Path,
    *,
    generation_id: str,
    contract_sha256: str,
    publication_private: Path,
) -> None:
    event = sign_snapshot_publication_event(
        {
            "schema_version": "lt_pointer_publication_event.v3",
            "publication_domain_sha256": publication_domain_sha256(),
            "operation_id": _operation_id(1),
            "revision": 1,
            "previous_event_id": None,
            "previous_pointer_sha256": None,
            "generation_id": generation_id,
            "contract_sha256": contract_sha256,
        },
        private_key_path=publication_private,
    )
    event_path = data_root / "views" / "pfc_lt" / "events" / f"{event['event_id']}.json"
    _write_bytes(event_path, canonical_json(event))
    pointer = pointer_mapping_from_event(event)
    pointer_payload = canonical_json(pointer)
    anchor = sign_snapshot_publication_anchor(
        {
            "schema_version": "lt_pointer_publication_anchor.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "sequence": 1,
            "event_id": event["event_id"],
            "pointer_sha256": _sha256(pointer_payload),
            "previous_anchor_id": None,
        },
        private_key_path=publication_private,
    )
    anchor_path = publication_anchor_directory(data_root, create=True) / (
        f"000000000001-{anchor['anchor_id']}.json"
    )
    _write_bytes(anchor_path, canonical_json(anchor))
    _write_bytes(data_root / "views" / "pfc_lt" / "current.json", pointer_payload)


def _unsigned_snapshot(
    tmp_path: Path,
    *,
    timestamp_private: Path,
    journal_private: Path,
) -> tuple[Path, dict[str, object]]:
    data_root = tmp_path / "data-root"
    generation = data_root / "snapshots" / "generation-001"
    policy_sha = _sha256(b"strict-quality-policy-v1")
    files: dict[str, object] = {}
    previous: str | None = None
    for sequence, (role, source_system) in enumerate(SOURCE_SYSTEMS.items(), start=1):
        received = f"2026-07-16T07:{sequence:02d}:00Z"
        available = f"2026-07-16T08:{sequence:02d}:00Z"
        raw_frame = _bronze_frame(role)
        provider_payload, provider_start, provider_end, source_locator = _provider_envelope(
            role, raw_frame, received_at_utc=received
        )
        provider_raw_path = generation / "provider_raw" / f"{role}.json"
        _write_bytes(provider_raw_path, provider_payload)
        provider_parser_path = approved_provider_parser_path_for_role(role)
        provider_parser_code = provider_parser_path.read_bytes()
        archived_provider_parser_path = generation / "provider_parser" / f"{role}.py"
        _write_bytes(archived_provider_parser_path, provider_parser_code)
        provider_config = json.dumps(
            build_provider_transform_config(
                role=role,
                start_utc=provider_start,
                end_utc=provider_end,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        provider_config_path = generation / "provider_parser" / f"{role}.json"
        _write_bytes(provider_config_path, provider_config)
        pd.testing.assert_frame_equal(
            transform_provider_raw(
                envelope_payload=provider_payload,
                parser_config=json.loads(provider_config),
            ),
            raw_frame,
            check_freq=False,
        )
        derived_frame = _derive_frame(role, raw_frame)
        raw_payload = _parquet_payload(raw_frame)
        raw_path = generation / "raw" / f"{role}.parquet"
        _write_bytes(raw_path, raw_payload)
        derived_payload = _parquet_payload(derived_frame)
        derived_path = generation / "inputs" / f"{role}.parquet"
        _write_bytes(derived_path, derived_payload)
        approved_parser = approved_parser_path_for_role(role)
        parser_code = approved_parser.read_bytes()
        parser_code_path = generation / "parser" / f"{role}.py"
        _write_bytes(parser_code_path, parser_code)
        parser_config = json.dumps(
            build_replay_config(
                role=role,
                source_system=source_system,
                raw_frame=raw_frame,
                derived_frame=derived_frame,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        parser_config_path = generation / "parser" / f"{role}.json"
        _write_bytes(parser_config_path, parser_config)
        receipt = sign_source_acquisition_receipt(
            {
                "schema_version": "source_acquisition_receipt.v2",
                "journal_id": "journal-001",
                "sequence": sequence,
                "previous_receipt_id": previous,
                "acquisition_id": "acquisition-001",
                "source_role": role,
                "source_system": source_system,
                "source_locator": source_locator,
                "acquisition_method": "PROSPECTIVE_DIRECT",
                "received_at_utc": received,
                "raw_sha256": _sha256(provider_payload),
                "raw_size_bytes": len(provider_payload),
            },
            private_key_path=timestamp_private,
        )
        previous = str(receipt["receipt_id"])
        entry = {
            "path": derived_path.relative_to(generation).as_posix(),
            "size_bytes": len(derived_payload),
            "sha256": _sha256(derived_payload),
            "acquisition_id": "acquisition-001",
            "available_at_utc": available,
            "calibration_eligible": True,
            "expected_cadence_seconds": 604800 if role == "hydro" else 900,
            "source_system": source_system,
            "source_receipt": receipt,
            "provider_raw_artifact": {
                "path": provider_raw_path.relative_to(generation).as_posix(),
                "size_bytes": len(provider_payload),
                "sha256": _sha256(provider_payload),
            },
            "provider_derivation": {
                "parser_code_path": archived_provider_parser_path.relative_to(
                    generation
                ).as_posix(),
                "parser_code_sha256": _sha256(provider_parser_code),
                "parser_config_path": provider_config_path.relative_to(generation).as_posix(),
                "parser_config_sha256": _sha256(provider_config),
                "derived_at_utc": available,
            },
            "raw_artifact": {
                "path": raw_path.relative_to(generation).as_posix(),
                "size_bytes": len(raw_payload),
                "sha256": _sha256(raw_payload),
            },
            "derivation": {
                "parser_code_path": parser_code_path.relative_to(generation).as_posix(),
                "parser_code_sha256": _sha256(parser_code),
                "parser_config_path": parser_config_path.relative_to(generation).as_posix(),
                "parser_config_sha256": _sha256(parser_config),
                "derived_at_utc": available,
            },
        }
        quality_payload = (
            json.dumps(
                {
                    "bindings": {
                        "provider_raw_sha256": _sha256(provider_payload),
                        "provider_parser_code_sha256": _sha256(provider_parser_code),
                        "provider_parser_config_sha256": _sha256(provider_config),
                        "bronze_sha256": _sha256(raw_payload),
                        "feature_parser_code_sha256": _sha256(parser_code),
                        "feature_parser_config_sha256": _sha256(parser_config),
                        "derived_sha256": _sha256(derived_payload),
                    },
                    "metrics": {
                        "bronze": _frame_quality_metrics(raw_frame),
                        "derived": _frame_quality_metrics(derived_frame),
                    },
                    "policy_sha256": policy_sha,
                    "role": role,
                    "schema_version": "lt_source_quality.v2",
                    "status": "PASS",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        quality_path = generation / "quality" / f"{role}.json"
        _write_bytes(quality_path, quality_payload)
        entry["quality_evidence"] = {
            "status": "PASS",
            "policy_sha256": policy_sha,
            "report_path": quality_path.relative_to(generation).as_posix(),
            "report_sha256": _sha256(quality_payload),
        }
        files[role] = entry
    receipt_ids = [str(files[role]["source_receipt"]["receipt_id"]) for role in SOURCE_SYSTEMS]
    checkpoint = sign_source_journal_checkpoint(
        {
            "schema_version": "source_journal_checkpoint.v1",
            "journal_id": "journal-001",
            "acquisition_id": "acquisition-001",
            "first_sequence": 1,
            "last_sequence": len(receipt_ids),
            "previous_receipt_id": None,
            "receipt_ids": receipt_ids,
            "head_receipt_id": receipt_ids[-1],
            "issued_at_utc": "2026-07-16T08:05:00Z",
            "bundle_root_sha256": governed_snapshot_bundle_root_sha256(files),
        },
        private_key_path=journal_private,
    )
    return data_root, {
        "schema_version": PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
        "layout": "external_v2",
        "generation_id": "generation-001",
        "acquisition_id": "acquisition-001",
        "available_at_utc": "2026-07-16T08:05:00Z",
        "calibration_eligible": True,
        "source_class": "GOVERNED_ACQUISITION",
        "source_journal_checkpoint": checkpoint,
        "files": files,
    }


def _resign_snapshot_after_file_change(unsigned: dict[str, object]) -> dict[str, object]:
    checkpoint = dict(unsigned["source_journal_checkpoint"])
    checkpoint.pop("journal_attestation")
    receipt_ids = [
        str(unsigned["files"][role]["source_receipt"]["receipt_id"]) for role in SOURCE_SYSTEMS
    ]
    checkpoint["first_sequence"] = 1
    checkpoint["last_sequence"] = len(receipt_ids)
    checkpoint["receipt_ids"] = receipt_ids
    checkpoint["head_receipt_id"] = receipt_ids[-1]
    checkpoint["bundle_root_sha256"] = governed_snapshot_bundle_root_sha256(unsigned["files"])
    unsigned["source_journal_checkpoint"] = sign_source_journal_checkpoint(
        checkpoint,
        private_key_path=os.environ["TEST_JOURNAL_PRIVATE_KEY"],
    )
    return sign_acquisition_contract(
        unsigned,
        private_key_path=os.environ["TEST_ACQUISITION_PRIVATE_KEY"],
    )


def _update_quality_binding(
    generation: Path,
    entry: dict[str, object],
    *,
    binding: str,
    value: str,
) -> None:
    quality = entry["quality_evidence"]
    quality_path = generation / quality["report_path"]
    report = json.loads(quality_path.read_bytes())
    report["bindings"][binding] = value
    payload = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    quality_path.write_bytes(payload)
    quality["report_sha256"] = _sha256(payload)


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


def _bronze_frame(role: str) -> pd.DataFrame:
    if role in {"epex_ch", "epex_de", "epex_at", "epex_fr", "epex_it"}:
        index = pd.date_range("2026-06-01", periods=96, freq="15min", tz="UTC")
        frame = pd.DataFrame(
            {"price_eur_mwh": [60.0 + float(value % 24) for value in range(96)]},
            index=index,
        )
        frame.attrs["lt_observation_resolution_provenance"] = {
            "schema_version": "lt_observation_resolution_provenance.v2",
            "role": role,
            "source_cadence_seconds": 900,
            "output_cadence_seconds": 900,
            "source_observation_count": 96,
            "output_observation_count": 96,
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
        return frame
    if role == "entso":
        index = pd.date_range("2026-06-01", periods=8, freq="15min", tz="UTC")
        return pd.DataFrame(
            {
                "load_mw": [7000.0, 7020.0, 7040.0, 7060.0, 7080.0, 7100.0, 7120.0, 7140.0],
                "solar_mw": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                "wind_mw": [100.0, 110.0, 90.0, 120.0, 80.0, 130.0, 70.0, 140.0],
            },
            index=index,
        )
    if role == "hydro":
        index = pd.date_range(
            end=pd.Timestamp("2026-06-22", tz="Europe/Zurich"),
            periods=312,
            freq="W-MON",
        ).tz_convert("UTC")
        fill_pct = [
            42.0 + float(position % 52) * 0.2 + float(position // 52) * 0.5
            for position in range(len(index))
        ]
        return pd.DataFrame(
            {
                "fill_pct": fill_pct,
                "fill_gwh": [value * 100.0 for value in fill_pct],
                "max_capacity_gwh": [10000.0] * len(index),
                "uebrig_ch_gwh": [value * 20.0 for value in fill_pct],
                "wallis_gwh": [value * 40.0 for value in fill_pct],
                "graubuenden_gwh": [value * 25.0 for value in fill_pct],
                "tessin_gwh": [value * 15.0 for value in fill_pct],
            },
            index=index,
        )
    raise AssertionError(role)


def _provider_envelope(
    role: str,
    frame: pd.DataFrame,
    *,
    received_at_utc: str,
) -> tuple[bytes, str, str, str]:
    start = frame.index.min()
    if role == "hydro":
        end = frame.index.max() + pd.Timedelta(days=1)
    else:
        end = frame.index.max() + pd.Timedelta(minutes=15)
    start_utc = start.isoformat().replace("+00:00", "Z")
    end_utc = end.isoformat().replace("+00:00", "Z")
    if role.startswith("epex_"):
        zone = {
            "epex_ch": "CH",
            "epex_de": "DE-LU",
            "epex_at": "AT",
            "epex_fr": "FR",
            "epex_it": "IT-North",
        }[role]
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
            acquisition_id="acquisition-001",
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


def _derive_frame(role: str, raw: pd.DataFrame) -> pd.DataFrame:
    if role in {"epex_ch", "epex_de", "epex_at", "epex_fr", "epex_it"}:
        return ingest_epex._clean(raw)
    if role == "entso":
        return ingest_entso.build_features(raw)
    if role == "hydro":
        return ingest_hydro.build_water_value(raw)
    raise AssertionError(role)


def _parquet_payload(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=True)
    return buffer.getvalue()


def _write_keypair(
    root: Path,
    private_key: Ed25519PrivateKey,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    private_path = root / "private.pem"
    public_path = root / "public.pem"
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
    return private_path, public_path


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"),
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _operation_id(sequence: int) -> str:
    return f"00000000-0000-4000-8000-{sequence:012d}"


def _second_generation(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    contract = json.loads((destination / "lt_input_snapshot.json").read_text(encoding="utf-8"))
    contract.pop("acquisition_attestation")
    contract["generation_id"] = "generation-002"
    _write_json(
        destination / "lt_input_snapshot.json",
        sign_acquisition_contract(
            contract,
            private_key_path=os.environ["TEST_ACQUISITION_PRIVATE_KEY"],
        ),
    )
    return destination


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
