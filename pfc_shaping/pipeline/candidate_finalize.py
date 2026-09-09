#!/usr/bin/env python3
"""Compatibility finalizer facade; canonical CLI is run_governed_lt_release finalize."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pfc_shaping.pipeline.atomic_promotion import (
    CandidateBundle,
    finalize_assembled_candidate_staging,
    validate_release_run_id,
)
from pfc_shaping.pipeline.candidate_evidence import (
    seal_assembled_candidate_evidence,
)
from pfc_shaping.pipeline.candidate_product_evidence import (
    STAGED_POLICY,
    CandidateProductEvidenceError,
    assemble_candidate_product_evidence,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    add_finalize_arguments,
    assert_installed_runtime_sealed,
    assert_namespace_absolute_paths,
    assert_phase_private_key_scope,
)


def finalize_candidate(args: argparse.Namespace) -> dict[str, object]:
    """Apply independent policy evidence and finalize one staged candidate."""

    assert_installed_runtime_sealed()
    assert_phase_private_key_scope("finalize")
    assert_namespace_absolute_paths(
        args,
        ("release_root", "failure_root", "source_hierarchy_policy"),
    )
    validate_release_run_id(args.run_id)
    candidates = args.release_root / "candidates"
    staging = candidates / f".{args.run_id}.staging"
    final = candidates / args.run_id
    if final.is_dir():
        bundle = finalize_assembled_candidate_staging(
            args.release_root,
            run_id=args.run_id,
        )
        _verify_retry_policy(bundle.path, args.source_hierarchy_policy)
        return _result(bundle)
    assemble_candidate_product_evidence(
        staging,
        run_id=args.run_id,
        source_hierarchy_policy_path=args.source_hierarchy_policy,
    )
    seal_assembled_candidate_evidence(staging, run_id=args.run_id)
    bundle = finalize_assembled_candidate_staging(
        args.release_root,
        run_id=args.run_id,
    )
    return _result(bundle)


def _verify_retry_policy(bundle_path: Path, supplied_policy: Path) -> None:
    try:
        selected = Path(supplied_policy)
        if selected.is_symlink():
            raise OSError("policy link is forbidden")
        supplied = selected.resolve(strict=True)
        if not supplied.is_file():
            raise OSError("policy is not a regular file")
        before = supplied.stat()
        supplied_bytes = supplied.read_bytes()
        after = supplied.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError("policy changed while being read")
        staged_bytes = (bundle_path / STAGED_POLICY).read_bytes()
    except OSError as exc:
        raise CandidateProductEvidenceError(
            "finalize retry source hierarchy policy is unavailable"
        ) from exc
    if supplied_bytes != staged_bytes:
        raise CandidateProductEvidenceError(
            "finalize retry source hierarchy policy differs from sealed candidate"
        )


def _result(bundle: CandidateBundle) -> dict[str, object]:
    return {
        "schema_version": "governed_lt_finalize_result.v1",
        "status": "CANDIDATE_FINALIZED_NOT_PROMOTED",
        "run_id": bundle.run_id,
        "candidate_path": str(bundle.path),
        "candidate_bundle_manifest_sha256": bundle.manifest_sha256,
        "promotion_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_finalize_arguments(parser)
    args = parser.parse_args(argv)
    assert_phase_private_key_scope("finalize")
    print(json.dumps(finalize_candidate(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
