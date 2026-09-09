"""Anchor-only signing capability excluded from publisher and model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pfc_shaping.data.snapshot_publication_contract import (
    _HEAD_OBSERVATION_FIELDS,
    _RECEIPT_FIELDS,
    PUBLICATION_HEAD_OBSERVATION_SCHEMA,
    PUBLICATION_RECEIPT_SCHEMA,
    _sign_document,
)


def sign_snapshot_anchor_receipt(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    return _sign_document(
        payload,
        schema=PUBLICATION_RECEIPT_SCHEMA,
        identity_field="receipt_id",
        expected_fields=_RECEIPT_FIELDS,
        private_key_path=private_key_path,
    )


def sign_snapshot_anchor_head_observation(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    return _sign_document(
        payload,
        schema=PUBLICATION_HEAD_OBSERVATION_SCHEMA,
        identity_field="observation_id",
        expected_fields=_HEAD_OBSERVATION_FIELDS,
        private_key_path=private_key_path,
    )
