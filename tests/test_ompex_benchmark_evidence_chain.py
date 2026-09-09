from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.validation.ompex_benchmark_evidence_chain import (
    AUTHORITY_CLASSIFICATION,
    CANDIDATE_DOMAIN,
    CANDIDATE_FREEZE_SCHEMA,
    OMPEX_AVAILABILITY_SCHEMA,
    OMPEX_DOMAIN,
    SIGNATURE_ALGORITHM,
    TRUTH_DOMAIN,
    TRUTH_PUBLICATION_SCHEMA,
    OmpexBenchmarkEvidenceChainError,
    canonical_json,
    public_key_id,
    receipt_id,
    signature_payload,
    verify_reference_evidence_chain,
)
from scripts.audit_ompex_benchmark_evidence_chain import audit

SHA = {
    "origin": "1" * 64,
    "commitment": "2" * 64,
    "candidate": "3" * 64,
    "config": "4" * 64,
    "inputs": "5" * 64,
    "ompex": "6" * 64,
    "timestamp": "7" * 64,
    "truth": "8" * 64,
}
ORIGIN = "2026-08-01T10:00:00Z"
TARGET_START = "2026-09-01T00:00:00Z"
TARGET_END = "2026-10-01T00:00:00Z"


def _common(*, schema: str, role: str, signer_role: str) -> dict[str, object]:
    return {
        "schema_version": schema,
        "authority_classification": AUTHORITY_CLASSIFICATION,
        "evidence_role": role,
        "test_fixture": True,
        "origin_id": SHA["origin"],
        "origin_utc": ORIGIN,
        "target_start_utc": TARGET_START,
        "target_end_utc": TARGET_END,
        "signer_organization_role": signer_role,
    }


def _sign(
    core: Mapping[str, object], *, domain: bytes, key: Ed25519PrivateKey
) -> tuple[dict[str, object], bytes]:
    identifier = receipt_id(domain=domain, core=core)
    unsigned = {**core, "receipt_id": identifier}
    signature = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": public_key_id(key.public_key()),
        "value_base64": base64.b64encode(
            key.sign(signature_payload(domain=domain, unsigned_receipt=unsigned))
        ).decode("ascii"),
    }
    document = {**unsigned, "signature": signature}
    return document, canonical_json(document)


def _chain(
    *,
    candidate_changes: Mapping[str, object] | None = None,
    ompex_changes: Mapping[str, object] | None = None,
    truth_changes: Mapping[str, object] | None = None,
    reuse_candidate_key_for_ompex: bool = False,
) -> dict[str, object]:
    candidate_key = Ed25519PrivateKey.generate()
    ompex_key = (
        candidate_key if reuse_candidate_key_for_ompex else Ed25519PrivateKey.generate()
    )
    truth_key = Ed25519PrivateKey.generate()

    candidate_core = {
        **_common(
            schema=CANDIDATE_FREEZE_SCHEMA,
            role="candidate_freeze",
            signer_role="independent_candidate_registry_reference",
        ),
        "candidate_commitment_id": "9" * 64,
        "candidate_commitment_sha256": SHA["commitment"],
        "candidate_artifact_sha256": SHA["candidate"],
        "candidate_configuration_manifest_sha256": SHA["config"],
        "candidate_input_manifest_sha256": SHA["inputs"],
        "data_cutoff_utc": ORIGIN,
        "frozen_at_utc": "2026-08-01T10:01:00Z",
        "ompex_opened_before_freeze": False,
        "truth_opened_before_freeze": False,
        **dict(candidate_changes or {}),
    }
    candidate, candidate_payload = _sign(
        candidate_core, domain=CANDIDATE_DOMAIN, key=candidate_key
    )

    ompex_core = {
        **_common(
            schema=OMPEX_AVAILABILITY_SCHEMA,
            role="ompex_availability",
            signer_role="desk_or_vendor_availability_reference",
        ),
        "candidate_freeze_receipt_id": candidate["receipt_id"],
        "filename": "HFC_Ompex_20260801_101700.xlsx",
        "source_size_bytes": 123456,
        "ompex_sha256": SHA["ompex"],
        "first_available_at_utc": "2026-08-01T09:59:00Z",
        "observed_at_utc": "2026-08-01T10:02:00Z",
        "opened_at_utc": "2026-08-01T10:02:00Z",
        "timestamp_semantics_contract_sha256": SHA["timestamp"],
        "asserted_hour_end_semantics_authenticated": True,
        "asserted_availability_at_origin_authenticated": True,
        "truth_opened_before_observation": False,
        **dict(ompex_changes or {}),
    }
    ompex, ompex_payload = _sign(ompex_core, domain=OMPEX_DOMAIN, key=ompex_key)

    truth_core = {
        **_common(
            schema=TRUTH_PUBLICATION_SCHEMA,
            role="truth_publication",
            signer_role="independent_truth_publication_reference",
        ),
        "candidate_freeze_receipt_id": candidate["receipt_id"],
        "ompex_availability_receipt_id": ompex["receipt_id"],
        "truth_source_id": "CH_REALISED_PRICE_REFERENCE",
        "source_size_bytes": 654321,
        "truth_sha256": SHA["truth"],
        "native_resolution": "PT1H",
        "timezone": "UTC",
        "currency_unit": "EUR/MWh",
        "published_at_utc": "2026-10-01T00:05:00Z",
        "opened_at_utc": "2026-10-01T00:06:00Z",
        "independent_of_candidate": True,
        "independent_of_ompex": True,
        "revision_status": "FROZEN_REGISTERED_REVISION",
        **dict(truth_changes or {}),
    }
    truth, truth_payload = _sign(truth_core, domain=TRUTH_DOMAIN, key=truth_key)
    return {
        "candidate_key": candidate_key,
        "ompex_key": ompex_key,
        "truth_key": truth_key,
        "candidate": candidate,
        "ompex": ompex,
        "truth": truth,
        "candidate_payload": candidate_payload,
        "ompex_payload": ompex_payload,
        "truth_payload": truth_payload,
    }


def _verify(chain: Mapping[str, object]) -> dict[str, object]:
    candidate_payload = chain["candidate_payload"]
    ompex_payload = chain["ompex_payload"]
    truth_payload = chain["truth_payload"]
    assert isinstance(candidate_payload, bytes)
    assert isinstance(ompex_payload, bytes)
    assert isinstance(truth_payload, bytes)
    return verify_reference_evidence_chain(
        candidate_receipt_payload=candidate_payload,
        ompex_receipt_payload=ompex_payload,
        truth_receipt_payload=truth_payload,
        candidate_public_key=chain["candidate_key"].public_key(),
        ompex_public_key=chain["ompex_key"].public_key(),
        truth_public_key=chain["truth_key"].public_key(),
        expected_candidate_receipt_sha256=hashlib.sha256(candidate_payload).hexdigest(),
        expected_ompex_receipt_sha256=hashlib.sha256(ompex_payload).hexdigest(),
        expected_truth_receipt_sha256=hashlib.sha256(truth_payload).hexdigest(),
        expected_candidate_commitment_id="9" * 64,
        expected_candidate_commitment_sha256=SHA["commitment"],
        expected_ompex_sha256=SHA["ompex"],
        expected_timestamp_semantics_contract_sha256=SHA["timestamp"],
        expected_truth_sha256=SHA["truth"],
        expected_origin_id=SHA["origin"],
        expected_origin_utc=ORIGIN,
        expected_target_start_utc=TARGET_START,
        expected_target_end_utc=TARGET_END,
    )


def test_valid_reference_chain_is_never_countable_or_scientific() -> None:
    assessment = _verify(_chain())

    assert assessment["status"] == (
        "PASS_LOCAL_REFERENCE_STRUCTURE_ONLY_NO_COUNTABLE_ORIGIN"
    )
    assert assessment["role_keys_distinct"] is True
    assert assessment["price_values_opened"] is False
    assert assessment["countable_origin_count"] == 0
    assert assessment["external_authority_verified"] is False
    assert assessment["scientific_scoring_authorized"] is False
    assert assessment["model_selection_authorized"] is False
    assert assessment["promotion_authorized"] is False
    assert assessment["production_authorized"] is False
    assert all(assessment["chronology"].values())


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        (
            "candidate",
            {"data_cutoff_utc": "2026-08-01T10:00:01Z"},
            "candidate data cutoff/freeze chronology",
        ),
        (
            "candidate",
            {"frozen_at_utc": "2026-09-01T00:00:01Z"},
            "candidate data cutoff/freeze chronology",
        ),
        (
            "candidate",
            {"frozen_at_utc": "2026-08-01T09:59:59Z"},
            "candidate data cutoff/freeze chronology",
        ),
        (
            "ompex",
            {"first_available_at_utc": "2026-08-01T10:00:01Z"},
            "OMPEX observation chronology",
        ),
        (
            "ompex",
            {
                "observed_at_utc": "2026-08-01T10:00:30Z",
                "opened_at_utc": "2026-08-01T10:00:30Z",
            },
            "OMPEX observation chronology",
        ),
        (
            "truth",
            {"published_at_utc": "2026-09-30T23:59:59Z"},
            "truth publication chronology",
        ),
        (
            "truth",
            {"opened_at_utc": "2026-10-01T00:04:59Z"},
            "truth publication chronology",
        ),
    ],
)
def test_chronology_fails_closed(
    section: str, changes: Mapping[str, object], message: str
) -> None:
    kwargs = {f"{section}_changes": changes}
    with pytest.raises(OmpexBenchmarkEvidenceChainError, match=message):
        _verify(_chain(**kwargs))


def test_role_keys_must_be_distinct() -> None:
    with pytest.raises(OmpexBenchmarkEvidenceChainError, match="distinct keys"):
        _verify(_chain(reuse_candidate_key_for_ompex=True))


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        (
            "candidate",
            {"candidate_commitment_id": "a" * 64},
            "candidate commitment ID binding",
        ),
        (
            "ompex",
            {"timestamp_semantics_contract_sha256": "b" * 64},
            "timestamp semantics contract binding",
        ),
    ],
)
def test_caller_held_identity_bindings_fail_closed(
    section: str, changes: Mapping[str, object], message: str
) -> None:
    with pytest.raises(OmpexBenchmarkEvidenceChainError, match=message):
        _verify(_chain(**{f"{section}_changes": changes}))


def test_cross_receipt_binding_and_signature_tamper_fail_closed() -> None:
    chain = _chain(ompex_changes={"candidate_freeze_receipt_id": "a" * 64})
    with pytest.raises(
        OmpexBenchmarkEvidenceChainError,
        match="OMPEX candidate-freeze receipt binding",
    ):
        _verify(chain)

    chain = _chain()
    truth = dict(chain["truth"])
    signature = dict(truth["signature"])
    value = str(signature["value_base64"])
    signature["value_base64"] = ("A" if value[0] != "A" else "B") + value[1:]
    truth["signature"] = signature
    chain["truth_payload"] = canonical_json(truth)
    with pytest.raises(OmpexBenchmarkEvidenceChainError, match="signature is invalid"):
        _verify(chain)


def test_noncanonical_receipt_and_truth_nonindependence_fail_closed() -> None:
    chain = _chain()
    chain["candidate_payload"] = chain["candidate_payload"] + b"\n"
    with pytest.raises(OmpexBenchmarkEvidenceChainError, match="not exact canonical JSON"):
        _verify(chain)

    with pytest.raises(OmpexBenchmarkEvidenceChainError, match="truth semantic boundary"):
        _verify(_chain(truth_changes={"independent_of_ompex": False}))


def test_local_runner_writes_only_noncountable_price_free_assessment(
    tmp_path,
) -> None:
    chain = _chain()
    paths = {}
    for role in ("candidate", "ompex", "truth"):
        receipt_path = tmp_path / f"{role}-receipt.json"
        receipt_path.write_bytes(chain[f"{role}_payload"])
        key_path = tmp_path / f"{role}-public.pem"
        key_path.write_bytes(
            chain[f"{role}_key"].public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        paths[f"{role}_receipt"] = receipt_path
        paths[f"{role}_key"] = key_path
    output = tmp_path / "assessment.json"

    assessment = audit(
        candidate_receipt=paths["candidate_receipt"],
        ompex_receipt=paths["ompex_receipt"],
        truth_receipt=paths["truth_receipt"],
        candidate_public_key=paths["candidate_key"],
        ompex_public_key=paths["ompex_key"],
        truth_public_key=paths["truth_key"],
        expected_candidate_receipt_sha256=hashlib.sha256(
            chain["candidate_payload"]
        ).hexdigest(),
        expected_ompex_receipt_sha256=hashlib.sha256(
            chain["ompex_payload"]
        ).hexdigest(),
        expected_truth_receipt_sha256=hashlib.sha256(
            chain["truth_payload"]
        ).hexdigest(),
        expected_candidate_commitment_id="9" * 64,
        expected_candidate_commitment_sha256=SHA["commitment"],
        expected_ompex_sha256=SHA["ompex"],
        expected_timestamp_semantics_contract_sha256=SHA["timestamp"],
        expected_truth_sha256=SHA["truth"],
        expected_origin_id=SHA["origin"],
        expected_origin_utc=ORIGIN,
        expected_target_start_utc=TARGET_START,
        expected_target_end_utc=TARGET_END,
        output_json=output,
    )

    assert output.exists()
    assert assessment["price_values_opened"] is False
    assert assessment["countable_origin_count"] == 0
    assert assessment["scientific_scoring_authorized"] is False
    source = __import__(
        "scripts.audit_ompex_benchmark_evidence_chain", fromlist=["ignored"]
    ).__file__
    assert source is not None
    assert "H:" + "\\" not in Path(source).read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        audit(
            candidate_receipt=paths["candidate_receipt"],
            ompex_receipt=paths["ompex_receipt"],
            truth_receipt=paths["truth_receipt"],
            candidate_public_key=paths["candidate_key"],
            ompex_public_key=paths["ompex_key"],
            truth_public_key=paths["truth_key"],
            expected_candidate_receipt_sha256=hashlib.sha256(
                chain["candidate_payload"]
            ).hexdigest(),
            expected_ompex_receipt_sha256=hashlib.sha256(
                chain["ompex_payload"]
            ).hexdigest(),
            expected_truth_receipt_sha256=hashlib.sha256(
                chain["truth_payload"]
            ).hexdigest(),
            expected_candidate_commitment_id="9" * 64,
            expected_candidate_commitment_sha256=SHA["commitment"],
            expected_ompex_sha256=SHA["ompex"],
            expected_timestamp_semantics_contract_sha256=SHA["timestamp"],
            expected_truth_sha256=SHA["truth"],
            expected_origin_id=SHA["origin"],
            expected_origin_utc=ORIGIN,
            expected_target_start_utc=TARGET_START,
            expected_target_end_utc=TARGET_END,
            output_json=output,
        )


def test_reference_contract_never_claims_external_or_decision_authority() -> None:
    contract_path = (
        Path(__file__).parents[1]
        / ".planning"
        / "phases"
        / "14-lt-audit-remediation"
        / "OMPEX-BENCHMARK-EVIDENCE-CHAIN-REFERENCE-CONTRACT-V1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["status"] == "NON_PRODUCTION_LOCAL_REFERENCE_NEVER_COUNTABLE"
    assert contract["trust_boundary"]["countable_origin_count"] == 0
    assert contract["trust_boundary"]["external_authority_verified"] is False
    assert contract["trust_boundary"]["trusted_time_verified"] is False
    assert contract["trust_boundary"]["price_values_opened_by_verifier"] is False
    assert all(value is False for value in contract["decision_authority"].values())
    assert len(contract["receipt_sequence"]) == 3
    assert len(contract["production_wire_requirements"]) >= 8
    assert contract["local_execution"]["h_drive_runtime_dependency"] is False
    assert contract["local_execution"]["databricks_request_required"] is False
