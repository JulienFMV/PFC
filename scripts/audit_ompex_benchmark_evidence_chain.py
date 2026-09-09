"""Audit a local non-production OMPEX benchmark receipt chain.

The runner reads only hash-bound receipt bytes and public keys below the
canonical workspace.  It never opens candidate, OMPEX or truth price data and
can emit only a non-countable reference assessment below ``build/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.validation.ompex_benchmark_evidence_chain import (
    canonical_json,
    verify_reference_evidence_chain,
)

_MAX_RECEIPT_BYTES = 64 * 1024
_MAX_PUBLIC_KEY_BYTES = 16 * 1024


def _workspace_root() -> Path:
    root = assert_absolute_path_has_no_links(Path(__file__).absolute().parents[1])
    if assert_absolute_path_has_no_links(Path.cwd().absolute()) != root:
        raise ValueError("evidence-chain audit requires the canonical workspace as cwd")
    return root


def _local_file(path: Path, *, root: Path, label: str, max_bytes: int) -> bytes:
    selected = assert_absolute_path_has_no_links(path.absolute())
    try:
        selected.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be below the canonical workspace") from exc
    return read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)


def _public_key(payload: bytes, *, label: str) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid PEM public key") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{label} is not Ed25519")
    return key


def audit(
    *,
    candidate_receipt: Path,
    ompex_receipt: Path,
    truth_receipt: Path,
    candidate_public_key: Path,
    ompex_public_key: Path,
    truth_public_key: Path,
    expected_candidate_receipt_sha256: str,
    expected_ompex_receipt_sha256: str,
    expected_truth_receipt_sha256: str,
    expected_candidate_commitment_id: str,
    expected_candidate_commitment_sha256: str,
    expected_ompex_sha256: str,
    expected_timestamp_semantics_contract_sha256: str,
    expected_truth_sha256: str,
    expected_origin_id: str,
    expected_origin_utc: str,
    expected_target_start_utc: str,
    expected_target_end_utc: str,
    output_json: Path,
) -> dict[str, object]:
    root = _workspace_root()
    candidate_payload = _local_file(
        candidate_receipt,
        root=root,
        label="candidate freeze receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    ompex_payload = _local_file(
        ompex_receipt,
        root=root,
        label="OMPEX availability receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    truth_payload = _local_file(
        truth_receipt,
        root=root,
        label="truth publication receipt",
        max_bytes=_MAX_RECEIPT_BYTES,
    )
    candidate_key = _public_key(
        _local_file(
            candidate_public_key,
            root=root,
            label="candidate authority public key",
            max_bytes=_MAX_PUBLIC_KEY_BYTES,
        ),
        label="candidate authority public key",
    )
    ompex_key = _public_key(
        _local_file(
            ompex_public_key,
            root=root,
            label="OMPEX authority public key",
            max_bytes=_MAX_PUBLIC_KEY_BYTES,
        ),
        label="OMPEX authority public key",
    )
    truth_key = _public_key(
        _local_file(
            truth_public_key,
            root=root,
            label="truth authority public key",
            max_bytes=_MAX_PUBLIC_KEY_BYTES,
        ),
        label="truth authority public key",
    )
    assessment = verify_reference_evidence_chain(
        candidate_receipt_payload=candidate_payload,
        ompex_receipt_payload=ompex_payload,
        truth_receipt_payload=truth_payload,
        candidate_public_key=candidate_key,
        ompex_public_key=ompex_key,
        truth_public_key=truth_key,
        expected_candidate_receipt_sha256=expected_candidate_receipt_sha256,
        expected_ompex_receipt_sha256=expected_ompex_receipt_sha256,
        expected_truth_receipt_sha256=expected_truth_receipt_sha256,
        expected_candidate_commitment_id=expected_candidate_commitment_id,
        expected_candidate_commitment_sha256=expected_candidate_commitment_sha256,
        expected_ompex_sha256=expected_ompex_sha256,
        expected_timestamp_semantics_contract_sha256=(
            expected_timestamp_semantics_contract_sha256
        ),
        expected_truth_sha256=expected_truth_sha256,
        expected_origin_id=expected_origin_id,
        expected_origin_utc=expected_origin_utc,
        expected_target_start_utc=expected_target_start_utc,
        expected_target_end_utc=expected_target_end_utc,
    )

    build_root = (root / "build").resolve()
    output = output_json.resolve(strict=False)
    try:
        output.relative_to(build_root)
    except ValueError as exc:
        raise ValueError("evidence-chain assessment output must be below build/") from exc
    if output.suffix.lower() != ".json":
        raise ValueError("evidence-chain assessment output must be JSON")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json(assessment) + b"\n")
    return assessment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--ompex-receipt", type=Path, required=True)
    parser.add_argument("--truth-receipt", type=Path, required=True)
    parser.add_argument("--candidate-public-key", type=Path, required=True)
    parser.add_argument("--ompex-public-key", type=Path, required=True)
    parser.add_argument("--truth-public-key", type=Path, required=True)
    parser.add_argument("--expected-candidate-receipt-sha256", required=True)
    parser.add_argument("--expected-ompex-receipt-sha256", required=True)
    parser.add_argument("--expected-truth-receipt-sha256", required=True)
    parser.add_argument("--expected-candidate-commitment-id", required=True)
    parser.add_argument("--expected-candidate-commitment-sha256", required=True)
    parser.add_argument("--expected-ompex-sha256", required=True)
    parser.add_argument("--expected-timestamp-semantics-contract-sha256", required=True)
    parser.add_argument("--expected-truth-sha256", required=True)
    parser.add_argument("--expected-origin-id", required=True)
    parser.add_argument("--expected-origin-utc", required=True)
    parser.add_argument("--expected-target-start-utc", required=True)
    parser.add_argument("--expected-target-end-utc", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        candidate_receipt=args.candidate_receipt,
        ompex_receipt=args.ompex_receipt,
        truth_receipt=args.truth_receipt,
        candidate_public_key=args.candidate_public_key,
        ompex_public_key=args.ompex_public_key,
        truth_public_key=args.truth_public_key,
        expected_candidate_receipt_sha256=args.expected_candidate_receipt_sha256,
        expected_ompex_receipt_sha256=args.expected_ompex_receipt_sha256,
        expected_truth_receipt_sha256=args.expected_truth_receipt_sha256,
        expected_candidate_commitment_id=args.expected_candidate_commitment_id,
        expected_candidate_commitment_sha256=(
            args.expected_candidate_commitment_sha256
        ),
        expected_ompex_sha256=args.expected_ompex_sha256,
        expected_timestamp_semantics_contract_sha256=(
            args.expected_timestamp_semantics_contract_sha256
        ),
        expected_truth_sha256=args.expected_truth_sha256,
        expected_origin_id=args.expected_origin_id,
        expected_origin_utc=args.expected_origin_utc,
        expected_target_start_utc=args.expected_target_start_utc,
        expected_target_end_utc=args.expected_target_end_utc,
        output_json=args.output_json,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
