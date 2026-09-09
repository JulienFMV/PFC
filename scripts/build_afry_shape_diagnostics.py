"""Build a restricted calendar-free AFRY shape diagnostic bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pfc_shaping.validation.afry_shape_diagnostics import (
    build_shape_diagnostic_bundle,
    verify_shape_diagnostic_bundle,
)

CANONICAL_WORKSPACE = Path(r"C:\Users\jbattaglia\PFC_LT")


def _require_canonical_workspace() -> Path:
    workspace = Path.cwd().resolve()
    if str(workspace) != str(CANONICAL_WORKSPACE):
        raise RuntimeError(
            f"AFRY diagnostic cwd must be exactly {CANONICAL_WORKSPACE}; got {workspace}"
        )
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    git_root = Path(result.stdout.strip()).resolve()
    if str(git_root) != str(CANONICAL_WORKSPACE):
        raise RuntimeError(
            f"AFRY diagnostic Git root must be exactly {CANONICAL_WORKSPACE}; got {git_root}"
        )
    return workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog-bundle", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("build/afry-shape-diagnostics"),
    )
    args = parser.parse_args(argv)
    workspace = _require_canonical_workspace()
    bundle = build_shape_diagnostic_bundle(
        workspace=workspace,
        source_catalog_bundle=args.source_catalog_bundle,
        output_root=args.output_root,
    )
    manifest = verify_shape_diagnostic_bundle(
        bundle,
        workspace=workspace,
        source_catalog_bundle=args.source_catalog_bundle,
    )
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "bundle_id": manifest["bundle_id"],
                "status": manifest["status"],
                "decision_state": manifest["decision_state"],
                "empirical_validation_status": manifest["admission"][
                    "empirical_validation_status"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
