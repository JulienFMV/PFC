"""Offline IT authority for one-shot LT legacy bootstrap authorization.

This module is deliberately separate from the snapshot publisher. Production
publisher environments must not contain the bootstrap private key or this
offline signing role.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pfc_shaping.data.snapshot_publication_contract import (
    _BOOTSTRAP_AUTHORIZATION_FIELDS,
    PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
    _sign_document,
)

PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_ENV = (
    "PFC_DATA_PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_PATH"
)


def required_bootstrap_signing_private_key_path() -> Path:
    value = str(os.environ.get(PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_ENV, "")).strip()
    if not value:
        raise ValueError(f"{PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_ENV} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"{PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_ENV} must be absolute"
        )
    return path


def sign_snapshot_bootstrap_authorization(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
) -> dict[str, object]:
    """Sign one exact authorization under the offline IT bootstrap identity."""

    return _sign_document(
        payload,
        schema=PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
        identity_field="authorization_id",
        expected_fields=_BOOTSTRAP_AUTHORIZATION_FIELDS,
        private_key_path=private_key_path,
    )
