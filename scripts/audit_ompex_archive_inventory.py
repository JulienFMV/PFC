"""Materialize a price-free, content-addressed OMPEX archive inventory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pfc_shaping.validation.ompex_archive_inventory import (  # noqa: E402
    capture_ompex_archive_inventory,
)

PREVIOUS_OBSERVATION = {
    "file_count": 353,
    "member_set_complete": False,
    "file_names": [
        "Template/HFC_Ompex_15min.xlsx",
        "Template_Comparatif.xlsx",
    ],
}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip().replace("/", "\\")).resolve()


def materialize(source_dir: Path, output_dir: Path) -> str:
    expected_root = Path(r"C:\Users\jbattaglia\PFC_LT")
    if Path.cwd().resolve() != expected_root or _git_root() != expected_root:
        raise ValueError("OMPEX inventory must run from the canonical workspace")
    output = output_dir.resolve()
    build_root = (expected_root / "build").resolve()
    if output == build_root or build_root not in output.parents:
        raise ValueError("OMPEX inventory output must be a fresh child of build/")
    if output.exists():
        raise ValueError("OMPEX inventory output must not already exist")

    audit = capture_ompex_archive_inventory(
        source_dir,
        previous_observation=PREVIOUS_OBSERVATION,
    )
    inventory_payload = {
        "schema_version": "fmv_ompex_archive_file_inventory.v1",
        "files": list(audit.files),
        "price_values_or_statistics_emitted": False,
    }
    inventory_bytes = _canonical(inventory_payload)
    summary_bytes = _canonical(audit.summary)
    bound_files = [
        expected_root / "pfc_shaping/validation/ompex_archive_inventory.py",
        expected_root / "scripts/audit_ompex_archive_inventory.py",
        expected_root / "tests/test_ompex_archive_inventory.py",
        expected_root
        / ".planning/phases/14-lt-audit-remediation/OMPEX-INDEPENDENT-BENCHMARK-CONTRACT-V1.json",
    ]
    components = {
        "source_files": {
            str(row["file_name"]): str(row["sha256"]) for row in audit.files
        },
        "bound_files": {
            str(path.relative_to(expected_root)).replace("\\", "/"): _hash_file(
                path
            )
            for path in bound_files
        },
        "inventory_sha256": sha256(inventory_bytes).hexdigest(),
        "summary_sha256": sha256(summary_bytes).hexdigest(),
    }
    content_id = sha256(_canonical(components)).hexdigest()
    manifest = {
        "schema_version": "fmv_ompex_archive_inventory_manifest.v1",
        "content_id": content_id,
        "components": components,
        "source_archive_copied": False,
        "benchmark_only": True,
        "rolling_origin_scoring_authorized": False,
        "price_values_or_statistics_emitted": False,
        "databricks_requests_issued": 0,
        "databricks_writes_issued": 0,
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "inventory.json").write_bytes(inventory_bytes)
    (output / "summary.json").write_bytes(summary_bytes)
    (output / "manifest.json").write_bytes(_canonical(manifest))
    return content_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(materialize(arguments.source_dir, arguments.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
