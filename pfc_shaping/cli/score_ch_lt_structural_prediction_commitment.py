"""Package CLI for one matured CH LT target against an exact commitment."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.validation.ch_lt_prospective_hourly_scoring import (
    ProspectiveHourlyScoringError,
    canonical_score_bytes,
    score_structural_commitment,
)

# Capture the governed invocation root once without embedding a workstation path.
ROOT = Path.cwd()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _directory_identity(path: Path) -> tuple[int, int]:
    selected = assert_absolute_path_has_no_links(path)
    metadata = selected.stat(follow_symlinks=False)
    if not selected.is_dir():
        raise ProspectiveHourlyScoringError("output parent is not a directory")
    return int(metadata.st_dev), int(metadata.st_ino)


def build_score(
    root: Path,
    output_directory: Path,
    *,
    commitment_path: Path,
    expected_commitment_sha256: str,
    predictions_path: Path,
    expected_predictions_sha256: str,
    scoring_contract_path: Path,
    expected_scoring_contract_sha256: str,
    truth_publication_receipt_path: Path,
    expected_truth_publication_receipt_sha256: str,
    delivery_month: str,
) -> dict[str, object]:
    root = assert_absolute_path_has_no_links(root)
    if root != ROOT or Path.cwd() != root:
        raise ProspectiveHourlyScoringError(
            "cwd and root must be the canonical repository"
        )
    output_directory = Path(os.path.abspath(output_directory))
    if not _inside(output_directory, root / "build"):
        raise ProspectiveHourlyScoringError("output must remain below build/")
    if output_directory.exists():
        raise ProspectiveHourlyScoringError("output directory already exists")
    if not output_directory.parent.is_dir():
        raise ProspectiveHourlyScoringError("output parent must already exist")
    parent_identity = _directory_identity(output_directory.parent)

    score_core = score_structural_commitment(
        root,
        commitment_path=commitment_path,
        expected_commitment_sha256=expected_commitment_sha256,
        predictions_path=predictions_path,
        expected_predictions_sha256=expected_predictions_sha256,
        scoring_contract_path=scoring_contract_path,
        expected_scoring_contract_sha256=expected_scoring_contract_sha256,
        truth_publication_receipt_path=truth_publication_receipt_path,
        expected_truth_publication_receipt_sha256=(
            expected_truth_publication_receipt_sha256
        ),
        delivery_month=delivery_month,
    )
    score_id = hashlib.sha256(canonical_score_bytes(score_core)).hexdigest()
    document = {**score_core, "score_id": score_id}
    payload = canonical_score_bytes(document)
    if _directory_identity(output_directory.parent) != parent_identity:
        raise ProspectiveHourlyScoringError("output parent identity changed during scoring")
    staging = output_directory.parent / f".{output_directory.name}.staging"
    if staging.exists():
        assert_absolute_path_has_no_links(staging)
        if not staging.is_dir():
            raise ProspectiveHourlyScoringError("score staging residue differs")
        staging_names = set(path.name for path in staging.iterdir())
        if not staging_names.issubset({"score.json"}):
            raise ProspectiveHourlyScoringError("score staging residue differs")
        if "score.json" in staging_names and (
            staging / "score.json"
        ).read_bytes() != payload:
            raise ProspectiveHourlyScoringError("score staging residue differs")
    else:
        staging.mkdir()
    assert_absolute_path_has_no_links(staging)
    write_durable_exact_bytes(
        staging / "score.json",
        payload,
        label="prospective hourly score",
    )
    if (staging / "score.json").read_bytes() != payload:
        raise ProspectiveHourlyScoringError("score staging verification differs")
    if _directory_identity(output_directory.parent) != parent_identity:
        raise ProspectiveHourlyScoringError("output parent identity changed before publish")
    os.rename(staging, output_directory)
    assert_absolute_path_has_no_links(output_directory)
    if (
        _directory_identity(output_directory.parent) != parent_identity
        or read_stable_single_link_file(
            output_directory / "score.json", label="published prospective hourly score"
        )
        != payload
    ):
        raise ProspectiveHourlyScoringError("published score verification differs")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--expected-commitment-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-predictions-sha256", required=True)
    parser.add_argument("--scoring-contract", type=Path, required=True)
    parser.add_argument("--expected-scoring-contract-sha256", required=True)
    parser.add_argument("--truth-publication-receipt", type=Path, required=True)
    parser.add_argument("--expected-truth-publication-receipt-sha256", required=True)
    parser.add_argument("--delivery-month", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = build_score(
        ROOT,
        args.output_directory,
        commitment_path=args.commitment,
        expected_commitment_sha256=args.expected_commitment_sha256,
        predictions_path=args.predictions,
        expected_predictions_sha256=args.expected_predictions_sha256,
        scoring_contract_path=args.scoring_contract,
        expected_scoring_contract_sha256=args.expected_scoring_contract_sha256,
        truth_publication_receipt_path=args.truth_publication_receipt,
        expected_truth_publication_receipt_sha256=(
            args.expected_truth_publication_receipt_sha256
        ),
        delivery_month=args.delivery_month,
    )
    print(canonical_score_bytes(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
