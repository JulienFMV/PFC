"""Verify CH LT preregistration supersession from a sealed installed runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
)
from pfc_shaping.validation.ch_lt_preregistration_supersession import (
    verify_preregistration_supersession,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    return parser


def _failure(status: str, error: Exception) -> int:
    print(
        json.dumps(
            {
                "status": status,
                "superseded_v1_usable": False,
                "successor_exists": False,
                "scientific_admission": False,
                "execution_authorized": False,
                "production_authorization": False,
                "promotion_gate": False,
                "error": str(error),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        return _failure("INVALID_GOVERNED_LT_RUNTIME", exc)

    args = _parser().parse_args(argv)
    if not args.evidence_root.is_absolute() or not args.registry.is_absolute():
        return _failure(
            "INVALID_CH_LT_PREREGISTRATION_SUPERSESSION",
            ValueError("evidence root and registry must be absolute paths"),
        )
    try:
        result = verify_preregistration_supersession(
            repo_root=args.evidence_root,
            registry_path=args.registry,
            expected_registry_sha256=args.expected_registry_sha256,
        )
    except (OSError, ValueError) as exc:
        return _failure("INVALID_CH_LT_PREREGISTRATION_SUPERSESSION", exc)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
