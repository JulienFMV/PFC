"""Materialize a governed, local-only AFRY scenario data catalog."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from pfc_shaping.data.afry_scenarios import AfryDataContractError, materialize_afry_catalog

CANONICAL_WORKSPACE = Path(r"C:\Users\jbattaglia\PFC_LT")


def _require_canonical_workspace() -> Path:
    workspace = Path.cwd().resolve()
    if str(workspace) != str(CANONICAL_WORKSPACE):
        raise AfryDataContractError(
            f"AFRY intake cwd must be exactly {CANONICAL_WORKSPACE}; got {workspace}"
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
        raise AfryDataContractError(
            f"AFRY intake Git root must be exactly {CANONICAL_WORKSPACE}; got {git_root}"
        )
    return workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        type=Path,
        required=True,
        help="Governed source registration JSON inside the workspace",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("build/afry-catalog"),
        help="Content-addressed publication root below build/",
    )
    args = parser.parse_args(argv)
    workspace = _require_canonical_workspace()
    output = materialize_afry_catalog(
        workspace=workspace,
        registration_path=args.registration,
        output_root=args.output_root,
    )
    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "bundle": str(output),
                "bundle_id": catalog["bundle_id"],
                "decision_state": catalog["decision_state"],
                "integrity_verdict": catalog["integrity_verdict"],
                "model_input_authorized": catalog["admission"]["model_input_authorized"],
                "production_authorized": catalog["admission"]["production_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
