"""Retired filesystem-publication signing excluded from governed artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pfc_shaping.data.snapshot_publication_contract import (
    _ANCHOR_FIELDS,
    _EVENT_FIELDS,
    PUBLICATION_ANCHOR_SCHEMA,
    PUBLICATION_EVENT_SCHEMA,
    _sign_document,
)


def sign_snapshot_publication_event(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    return _sign_document(
        payload,
        schema=PUBLICATION_EVENT_SCHEMA,
        identity_field="event_id",
        expected_fields=_EVENT_FIELDS,
        private_key_path=private_key_path,
    )


def sign_snapshot_publication_anchor(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    return _sign_document(
        payload,
        schema=PUBLICATION_ANCHOR_SCHEMA,
        identity_field="anchor_id",
        expected_fields=_ANCHOR_FIELDS,
        private_key_path=private_key_path,
    )
