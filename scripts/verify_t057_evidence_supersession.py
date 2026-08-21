"""Verify the fail-closed T057 evidence supersession registry and sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

REGISTRY_SCHEMA_VERSION = "t057_evidence_supersession_registry.v1"
CORRECTION_SCHEMA_VERSION = "t057_fold_integrity_correction.v2"
REGISTRY_RELATIVE_PATH = Path(
    ".planning/phases/14-lt-audit-remediation/T057-EVIDENCE-SUPERSESSION-REGISTRY.json"
)
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_EXPECTED_SUPERSEDED_PATHS = {
    "legacy_energy_charts_run_summary": (
        "output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724/"
        "energy_charts_locked_holdout_run_summary.json"
    ),
    "legacy_spot_backtest_summary": (
        "output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724/"
        "locked_holdout_runner/spot_backtest_summary.json"
    ),
    "legacy_duplicated_rolling_folds_csv": (
        "output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724/"
        "locked_holdout_runner/rolling_spot_profile_folds.csv"
    ),
    "superseded_local_quality_report": "output/phase14/t057_local_quality_report_20260724_v3.json",
    "superseded_local_quality_report_v1_json": "output/phase14/t057_local_quality_report_20260724.json",
    "superseded_local_quality_report_v1_markdown": "output/phase14/t057_local_quality_report_20260724.md",
    "superseded_local_quality_report_v2_json": "output/phase14/t057_local_quality_report_20260724_v2.json",
    "superseded_local_quality_report_v2_markdown": "output/phase14/t057_local_quality_report_20260724_v2.md",
    "superseded_local_quality_report_v3_markdown": "output/phase14/t057_local_quality_report_20260724_v3.md",
    "superseded_correction_v1": (
        "output/phase14/t057_fold_integrity_correction_20260724_run1/"
        "t057_fold_integrity_correction.json"
    ),
}


def canonical_registry_path() -> Path:
    return assert_absolute_path_has_no_links(Path(__file__).resolve().parent.parent / REGISTRY_RELATIVE_PATH)


def supersession_record_for_sha256(
    sha256: str,
    *,
    registry: Path | None = None,
) -> dict[str, Any] | None:
    registry_path = _absolute_path(registry or canonical_registry_path())
    payload = _read_json(registry_path, label="T057 supersession registry")
    _validate_registry_header(payload)
    selected = _sha256_value(sha256, label="queried evidence")
    for record in _records(payload):
        if record["sha256"] == selected:
            return {
                **record,
                "registry": str(registry_path),
                "authoritative_correction": payload["authoritative_correction"],
                "effective_claims": payload["effective_claims"],
            }
    return None


def verify_supersession_registry(
    *,
    registry: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    registry_path = _absolute_path(registry or canonical_registry_path())
    root = _absolute_path(repo_root or Path(__file__).resolve().parent.parent)
    registry_bytes = read_stable_single_link_file(
        registry_path,
        label="T057 supersession registry",
        max_bytes=_MAX_JSON_BYTES,
    )
    payload = _strict_json(registry_bytes, label="T057 supersession registry")
    _validate_registry_header(payload)
    records = _records(payload)
    observed_paths = {record["kind"]: record["path"].replace("\\", "/") for record in records}
    if observed_paths != _EXPECTED_SUPERSEDED_PATHS:
        missing = sorted(set(_EXPECTED_SUPERSEDED_PATHS) - set(observed_paths))
        unexpected = sorted(set(observed_paths) - set(_EXPECTED_SUPERSEDED_PATHS))
        mismatched = sorted(
            kind
            for kind in set(observed_paths) & set(_EXPECTED_SUPERSEDED_PATHS)
            if observed_paths[kind] != _EXPECTED_SUPERSEDED_PATHS[kind]
        )
        raise ValueError(
            f"T057 supersession registry inventory mismatch: missing={missing}, "
            f"unexpected={unexpected}, mismatched_paths={mismatched}"
        )
    if len({record["path"].casefold() for record in records}) != len(records) or len(
        {record["kind"] for record in records}
    ) != len(records):
        raise ValueError("T057 supersession registry contains duplicate evidence identities")
    verified_hashes: dict[str, str] = {}
    for record in records:
        path = _evidence_path(record["path"], root=root)
        observed = _file_sha256(path, label=f"superseded evidence {record['kind']}")
        if observed != record["sha256"]:
            raise ValueError(f"superseded evidence hash mismatch: {record['kind']}")
        verified_hashes[record["kind"]] = observed

    authority = _mapping(payload.get("authoritative_correction"), label="correction authority")
    correction_path = _evidence_path(authority.get("path"), root=root)
    correction_bytes = read_stable_single_link_file(
        correction_path,
        label="authoritative T057 correction",
        max_bytes=_MAX_JSON_BYTES,
    )
    correction_sha = hashlib.sha256(correction_bytes).hexdigest()
    if correction_sha != _sha256_value(str(authority.get("sha256", "")), label="correction"):
        raise ValueError("authoritative T057 correction hash mismatch")
    correction = _strict_json(correction_bytes, label="authoritative T057 correction")
    source_contract = _mapping(correction.get("source_contract"), label="correction source contract")
    if not (
        authority.get("schema_version") == CORRECTION_SCHEMA_VERSION
        and correction.get("schema_version") == CORRECTION_SCHEMA_VERSION
        and correction.get("status") == "T057_HISTORICAL_ROLLING_EVIDENCE_REVOKED"
        and correction.get("production_authorization") is False
        and correction.get("promotion_gate") is False
        and source_contract.get("canonical_sources_only") is True
        and source_contract.get("caller_anchors_match_canonical_registry") is True
        and source_contract.get("candidate_spot_and_config_cross_bound") is True
        and source_contract.get("metrics_recomputed_from_csv") is True
    ):
        raise ValueError("authoritative T057 correction contract is invalid")

    legacy = _mapping(correction.get("legacy_evidence"), label="correction legacy evidence")
    supersession = _mapping(correction.get("supersession"), label="correction supersession")
    required_links = {
        "legacy_energy_charts_run_summary": legacy.get("run_summary_sha256"),
        "legacy_spot_backtest_summary": legacy.get("spot_backtest_summary_sha256"),
        "legacy_duplicated_rolling_folds_csv": legacy.get("rolling_folds_csv_sha256"),
        "superseded_local_quality_report": supersession.get("superseded_quality_report_sha256"),
    }
    if any(verified_hashes.get(kind) != digest for kind, digest in required_links.items()):
        raise ValueError("authoritative correction is not cross-bound to every superseded artifact")

    claims = _mapping(payload.get("effective_claims"), label="effective claims")
    if not (
        claims.get("historical_distinct_fold_count") == 1
        and claims.get("historical_robustness_established") is False
        and claims.get("future_holdout_episode_count") == 1
        and claims.get("future_holdout_hours") == 336
        and claims.get("materiality_established") is False
        and claims.get("production_go") is False
    ):
        raise ValueError("T057 effective supersession claims are invalid")
    return {
        "schema_version": "t057_evidence_supersession_verification.v1",
        "status": "T057_SUPERSESSION_VERIFIED",
        "registry": str(registry_path),
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "authoritative_correction": str(correction_path),
        "authoritative_correction_sha256": correction_sha,
        "superseded_evidence_count": len(records),
        "effective_claims": claims,
        "production_authorization": False,
        "promotion_gate": False,
    }


def _validate_registry_header(payload: Mapping[str, Any]) -> None:
    if not (
        payload.get("schema_version") == REGISTRY_SCHEMA_VERSION
        and payload.get("status") == "ACTIVE_FAIL_CLOSED_SUPERSESSION"
        and payload.get("plan_id") == "t057_locked_t056_future_holdout"
        and payload.get("production_authorization") is False
        and payload.get("promotion_gate") is False
        and isinstance(payload.get("authoritative_correction"), dict)
        and isinstance(payload.get("effective_claims"), dict)
    ):
        raise ValueError("T057 supersession registry header is invalid")


def _records(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("superseded_evidence")
    if not isinstance(raw, list) or not raw:
        raise ValueError("T057 supersession registry has no evidence records")
    records: list[dict[str, str]] = []
    for item in raw:
        mapping = _mapping(item, label="superseded evidence record")
        kind = str(mapping.get("kind", "")).strip()
        path = str(mapping.get("path", "")).strip()
        digest = _sha256_value(str(mapping.get("sha256", "")), label=kind or "evidence")
        if not kind or not path:
            raise ValueError("superseded evidence record is incomplete")
        records.append({"kind": kind, "path": path, "sha256": digest})
    return records


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return _strict_json(
        read_stable_single_link_file(path, label=label, max_bytes=_MAX_JSON_BYTES),
        label=label,
    )


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_mapping,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    return _mapping(value, label=label)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _evidence_path(value: Any, *, root: Path) -> Path:
    selected = Path(str(value)).expanduser()
    if not selected.is_absolute():
        selected = root / selected
    return _absolute_path(selected)


def _absolute_path(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return assert_absolute_path_has_no_links(Path(selected.absolute()))


def _file_sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(
        read_stable_single_link_file(path, label=label, max_bytes=_MAX_ARTIFACT_BYTES)
    ).hexdigest()


def _sha256_value(value: str, *, label: str) -> str:
    selected = value.strip().lower()
    if len(selected) != 64 or any(character not in "0123456789abcdef" for character in selected):
        raise ValueError(f"{label} SHA-256 is invalid")
    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=canonical_registry_path())
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = verify_supersession_registry(registry=args.registry, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
