"""Point-in-time data contracts for LT quant shaping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import hashlib

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    feature_group: str
    required_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    available_at_column: str = "available_at"
    min_coverage: float = 1.0
    severity: str = "P0"
    latest_revision_policy: str = "asof"


@dataclass(frozen=True)
class CoverageRequirement:
    feature_group: str
    required_keys: pd.DataFrame
    min_coverage: float = 1.0


def _timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def assert_no_future_available_at(
    df: pd.DataFrame,
    spec: FeatureSpec,
    valuation_timestamp: object,
) -> None:
    required = set(spec.required_columns) | set(spec.key_columns) | {spec.available_at_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{spec.feature_group} missing required columns: {missing}")
    valuation = _timestamp(valuation_timestamp)
    available = pd.to_datetime(df[spec.available_at_column], utc=True)
    future = available > valuation
    if bool(future.any()):
        raise ValueError(
            f"{spec.feature_group} contains {int(future.sum())} rows with "
            f"{spec.available_at_column} after valuation_timestamp"
        )


def asof_feature_frame(
    df: pd.DataFrame,
    spec: FeatureSpec,
    valuation_timestamp: object,
) -> pd.DataFrame:
    """Return latest point-in-time rows per key at ``valuation_timestamp``."""

    required = set(spec.required_columns) | set(spec.key_columns) | {spec.available_at_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{spec.feature_group} missing required columns: {missing}")
    valuation = _timestamp(valuation_timestamp)
    out = df.copy()
    out[spec.available_at_column] = pd.to_datetime(out[spec.available_at_column], utc=True)
    out = out[out[spec.available_at_column] <= valuation]
    if out.empty:
        return out.reset_index(drop=True)
    out = out.sort_values([*spec.key_columns, spec.available_at_column])
    return out.groupby(list(spec.key_columns), as_index=False, sort=False).tail(1).reset_index(drop=True)


def coverage_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec,
    requirement: CoverageRequirement,
) -> pd.DataFrame:
    if requirement.feature_group != spec.feature_group:
        raise ValueError("coverage requirement feature_group does not match spec")
    missing_keys = sorted(set(spec.key_columns) - set(requirement.required_keys.columns))
    if missing_keys:
        raise ValueError(f"required_keys missing key columns: {missing_keys}")
    present = df[list(spec.key_columns)].drop_duplicates() if not df.empty else pd.DataFrame(columns=spec.key_columns)
    merged = requirement.required_keys[list(spec.key_columns)].drop_duplicates().merge(
        present.assign(_present=True),
        on=list(spec.key_columns),
        how="left",
    )
    merged["_present"] = merged["_present"].eq(True)
    required_rows = len(merged)
    present_rows = int(merged["_present"].sum())
    ratio = float(present_rows / required_rows) if required_rows else 1.0
    available = pd.to_datetime(df[spec.available_at_column], utc=True) if spec.available_at_column in df else pd.Series(dtype="datetime64[ns, UTC]")
    status = "PASS" if ratio >= max(spec.min_coverage, requirement.min_coverage) else "NO_GO"
    return pd.DataFrame(
        [
            {
                "feature_group": spec.feature_group,
                "present_rows": present_rows,
                "required_rows": required_rows,
                "coverage_ratio": ratio,
                "max_available_at": available.max() if not available.empty else pd.NaT,
                "status": status,
                "severity": spec.severity,
            }
        ]
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_record(
    path: Path,
    *,
    feature_group: str,
    available_at: object | None = None,
) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "feature_group": feature_group,
        "available_at": _timestamp(available_at).isoformat() if available_at is not None else None,
        "bytes": path.stat().st_size,
    }


def manifest_skeleton(
    *,
    run_id: str,
    git_sha: str,
    git_branch: str,
    valuation_timestamp: object,
    input_artifacts: list[Mapping[str, object]],
    data_vintages: list[Mapping[str, object]],
    qa_verdict: str = "NO_GO",
) -> dict[str, object]:
    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "git_branch": git_branch,
        "valuation_timestamp": _timestamp(valuation_timestamp).isoformat(),
        "input_artifacts": list(input_artifacts),
        "data_vintages": list(data_vintages),
        "qa_verdict": qa_verdict,
    }
    validate_run_manifest_schema(manifest)
    return manifest


def validate_run_manifest_schema(manifest: Mapping[str, object]) -> None:
    required = {
        "run_id",
        "git_sha",
        "git_branch",
        "valuation_timestamp",
        "input_artifacts",
        "data_vintages",
        "qa_verdict",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"run manifest missing required fields: {missing}")
    if manifest["qa_verdict"] not in {"PASS", "CONDITIONAL", "NO_GO"}:
        raise ValueError("qa_verdict must be PASS, CONDITIONAL, or NO_GO")
    valuation = _timestamp(manifest["valuation_timestamp"])
    for collection_name in ("input_artifacts", "data_vintages"):
        collection = manifest[collection_name]
        if not isinstance(collection, list):
            raise ValueError(f"{collection_name} must be a list")
        for artifact in collection:  # type: ignore[assignment]
            if not isinstance(artifact, Mapping):
                raise ValueError(f"{collection_name} entries must be mappings")
            if "sha256" not in artifact or not artifact["sha256"]:
                raise ValueError(f"{collection_name} artifact missing sha256")
            available_at = artifact.get("available_at")
            if available_at is not None and _timestamp(available_at) > valuation:
                raise ValueError(f"{collection_name} artifact available_at is after valuation_timestamp")
