"""Quantify paired-origin dependence and power without authorizing evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import assert_absolute_path_has_no_links
from pfc_shaping.validation.dependence_power_preflight import (
    audit_dependence_power_pilot,
)


def run_audit(
    *,
    fold_csv_path: Path,
    expected_fold_csv_sha256: str,
    source_summary_path: Path,
    expected_source_summary_sha256: str,
    candidate_model: str,
    benchmark_model: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    result = audit_dependence_power_pilot(
        fold_csv_path=fold_csv_path,
        expected_fold_csv_sha256=expected_fold_csv_sha256,
        source_summary_path=source_summary_path,
        expected_source_summary_sha256=expected_source_summary_sha256,
        candidate_model=candidate_model,
        benchmark_model=benchmark_model,
    )
    if output_path is not None:
        target = _canonical_output(output_path)
        if target.exists():
            raise ValueError("dependence/power preflight output already exists")
        write_durable_exact_bytes(
            target,
            _canonical_json_bytes(result) + b"\n",
            label="CH LT dependence/power preflight",
        )
    return result


def _canonical_output(path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    allowed = root / "output" / "phase14" / "ch_lt_dependence_power_preflight"
    candidate = path if path.is_absolute() else root / path
    candidate = assert_absolute_path_has_no_links(candidate)
    if candidate.parent != allowed:
        raise ValueError(
            "dependence/power preflight output must be directly inside the canonical namespace"
        )
    if candidate.suffix.lower() != ".json":
        raise ValueError("dependence/power preflight output must be JSON")
    if not allowed.is_dir():
        raise ValueError("dependence/power preflight output directory is unavailable")
    return candidate


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-csv", type=Path, required=True)
    parser.add_argument("--expected-fold-csv-sha256", required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--expected-source-summary-sha256", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--benchmark-model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("validate-diagnostic", "admit-scientific-design"),
        default="admit-scientific-design",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_audit(
            fold_csv_path=args.fold_csv,
            expected_fold_csv_sha256=args.expected_fold_csv_sha256,
            source_summary_path=args.source_summary,
            expected_source_summary_sha256=args.expected_source_summary_sha256,
            candidate_model=args.candidate_model,
            benchmark_model=args.benchmark_model,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "INVALID_DEPENDENCE_POWER_PREFLIGHT",
                    "scientific_admission": False,
                    "execution_authorized": False,
                    "production_authorization": False,
                    "promotion_gate": False,
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if args.mode == "validate-diagnostic" else 3


if __name__ == "__main__":
    raise SystemExit(main())
