from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from pfc_shaping.pipeline.quote_conflict_policy_contract import (
    QuoteConflictPolicyAuthenticationError,
    sign_quote_conflict_policy,
    verify_quote_conflict_policy,
)


def _key_pair(tmp_path: Path, name: str) -> tuple[Path, Path]:
    key = Ed25519PrivateKey.generate()
    private = tmp_path / f"{name}-private.pem"
    public = tmp_path / f"{name}-public.pem"
    private.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private, public


def _policy() -> dict[str, object]:
    return {
        "schema_version": "ch_quote_conflict_source_hierarchy_policy.v1",
        "market": "CH",
        "forward_snapshot_date": "2026-07-13",
        "production_approved": True,
        "quote_conflict_identity_hash": "a" * 64,
    }


def test_policy_requires_independent_trusted_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, public = _key_pair(tmp_path, "trusted")
    monkeypatch.setenv("PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH", str(public))

    signed = sign_quote_conflict_policy(_policy(), private_key_path=private)

    assert verify_quote_conflict_policy(signed) == _policy()


def test_unsigned_or_attacker_signed_policy_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trusted_private, trusted_public = _key_pair(tmp_path, "trusted")
    attacker_private, _attacker_public = _key_pair(tmp_path, "attacker")
    monkeypatch.setenv(
        "PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH",
        str(trusted_public),
    )

    with pytest.raises(QuoteConflictPolicyAuthenticationError, match="attestation is missing"):
        verify_quote_conflict_policy(_policy())
    forged = sign_quote_conflict_policy(_policy(), private_key_path=attacker_private)
    with pytest.raises(QuoteConflictPolicyAuthenticationError, match="key is not trusted"):
        verify_quote_conflict_policy(forged)
