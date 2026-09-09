#!/usr/bin/env python3
"""Sign one exact, short-lived LT legacy-bootstrap authorization offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pfc_shaping.data.snapshot_bootstrap_authority import (
    required_bootstrap_signing_private_key_path,
    sign_snapshot_bootstrap_authorization,
)
from pfc_shaping.data.snapshot_publication_state import canonical_json
from pfc_shaping.durable_artifact import (
    DurableArtifactExistsError,
    durable_artifact_durability_classification,
    write_durable_exact_bytes,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json


def sign_authorization(input_path: Path, output_path: Path) -> dict[str, object]:
    source = assert_absolute_path_has_no_links(input_path.expanduser())
    payload = read_stable_single_link_file(source, label="unsigned bootstrap authorization")
    document = load_strict_json(payload.decode("ascii"))
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise ValueError("unsigned bootstrap authorization must be canonical JSON")
    if "signature" in document or "authorization_id" in document:
        raise ValueError("bootstrap authorization input must be unsigned")
    signed = sign_snapshot_bootstrap_authorization(
        document,
        private_key_path=required_bootstrap_signing_private_key_path(),
    )
    signed_payload = canonical_json(signed)
    target = output_path.expanduser()
    if not target.is_absolute():
        raise ValueError("signed bootstrap authorization output must be absolute")
    try:
        target = write_durable_exact_bytes(
            target,
            signed_payload,
            label="signed bootstrap authorization",
            replace_existing_if=_bootstrap_output_is_noncanonical,
        )
    except DurableArtifactExistsError as exc:
        raise ValueError(
            "signed bootstrap authorization output contains another valid document"
        ) from exc
    return {
        "schema_version": "lt_snapshot_bootstrap_signing_result.v1",
        "status": "SIGNED_OFFLINE_NOT_CONSUMED",
        "authorization_id": signed["authorization_id"],
        "authorization_path": str(target),
        "authorization_sha256": hashlib.sha256(signed_payload).hexdigest(),
        "local_durability": durable_artifact_durability_classification(),
        "production_authorization": False,
    }


def _bootstrap_output_is_noncanonical(payload: bytes) -> bool:
    try:
        document = load_strict_json(payload.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return True
    return not isinstance(document, dict) or canonical_json(document) != payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(sign_authorization(args.input, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
