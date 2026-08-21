"""Publish an append-only revocation sidecar for the superseded T057 fold claim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from scripts.epex_lab_locked_holdout_policy import (
    T057_PLAN_ID,
    T057_PLAN_SHA256,
    canonical_t057_plan_path,
)

SCHEMA_VERSION = "t057_fold_integrity_correction.v2"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_CSV_BYTES = 64 * 1024 * 1024
_ORIGINAL_RUN_RELATIVE_PATH = Path(
    "output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724/"
    "energy_charts_locked_holdout_run_summary.json"
)
_ORIGINAL_RUN_SHA256 = "7b1f8613ffdf8c5771c6d493299fbeec5ac8fc15d136d3ed428282e4e081ffc7"
_SUPERSEDED_QUALITY_RELATIVE_PATH = Path("output/phase14/t057_local_quality_report_20260724_v3.json")
_SUPERSEDED_QUALITY_SHA256 = "d16787c8d28d4c0ddd0cfd346cda786b09385684c66a5663454af584fcfcd389"
_FORENSIC_SUMMARY_RELATIVE_PATH = Path(
    "output/phase14/t057_fold_integrity_forensic_replay_20260724_run1/spot_backtest_summary.json"
)
_FORENSIC_SUMMARY_SHA256 = "0102105bc2320e0df99882e4104616c281c5cdd9b6e10ce24a9f9a49ac102624"


def build_correction(
    *,
    original_run_summary: Path,
    expected_original_run_summary_sha256: str,
    superseded_quality_report: Path,
    expected_superseded_quality_report_sha256: str,
    forensic_backtest_summary: Path,
    expected_forensic_backtest_summary_sha256: str,
    output: Path,
) -> dict[str, Any]:
    contract = _canonical_source_contract()
    _require_contract_source(
        original_run_summary,
        expected_original_run_summary_sha256,
        expected_path=contract["original_run"],
        expected_sha256=str(contract["original_run_sha256"]),
        label="original T057 run summary",
    )
    _require_contract_source(
        superseded_quality_report,
        expected_superseded_quality_report_sha256,
        expected_path=contract["superseded_quality"],
        expected_sha256=str(contract["superseded_quality_sha256"]),
        label="superseded T057 quality report",
    )
    _require_contract_source(
        forensic_backtest_summary,
        expected_forensic_backtest_summary_sha256,
        expected_path=contract["forensic_summary"],
        expected_sha256=str(contract["forensic_summary_sha256"]),
        label="forensic T057 backtest summary",
    )
    original_path, original_bytes = _read_anchored(
        original_run_summary,
        expected_original_run_summary_sha256,
        label="original T057 run summary",
        max_bytes=_MAX_JSON_BYTES,
    )
    quality_path, quality_bytes = _read_anchored(
        superseded_quality_report,
        expected_superseded_quality_report_sha256,
        label="superseded T057 quality report",
        max_bytes=_MAX_JSON_BYTES,
    )
    forensic_path, forensic_bytes = _read_anchored(
        forensic_backtest_summary,
        expected_forensic_backtest_summary_sha256,
        label="forensic T057 backtest summary",
        max_bytes=_MAX_JSON_BYTES,
    )
    original = _strict_json(original_bytes, label="original T057 run summary")
    quality = _strict_json(quality_bytes, label="superseded T057 quality report")
    forensic = _strict_json(forensic_bytes, label="forensic T057 backtest summary")
    plan_path, plan_bytes = _read_anchored(
        contract["plan"],
        str(contract["plan_sha256"]),
        label="canonical frozen T057 plan",
        max_bytes=_MAX_JSON_BYTES,
    )
    plan = _strict_json(plan_bytes, label="canonical frozen T057 plan")

    if original.get("schema_version") != "energy_charts_epex_locked_holdout_run.v1":
        raise ValueError("correction source must be the legacy T057 wrapper v1")
    if quality.get("schema_version") != "pfc_local_quality_report.v1":
        raise ValueError("correction source must be the superseded quality report v1")
    if quality.get("status") != "LOCAL_DIAGNOSTIC_PASS_NOT_PRODUCTION":
        raise ValueError("superseded quality report status is unexpected")
    if forensic.get("schema_version") != "epex_shape_lab_spot_backtest.v2":
        raise ValueError("forensic correction must bind the legacy backtest v2 replay")
    if any(document.get("production_approved") is True for document in (original, forensic)):
        raise ValueError("correction evidence must remain non-production")
    locked = _mapping(original.get("locked_holdout"), label="legacy locked holdout")
    original_spot_path, original_spot_bytes = _read_linked(
        locked,
        path_key="spot_backtest_summary",
        sha_key="spot_backtest_summary_sha256",
        label="legacy spot backtest summary",
        max_bytes=_MAX_JSON_BYTES,
    )
    original_spot = _strict_json(original_spot_bytes, label="legacy spot backtest summary")
    _validate_canonical_cross_bindings(
        original=original,
        original_sha256=hashlib.sha256(original_bytes).hexdigest(),
        original_spot=original_spot,
        original_spot_sha256=hashlib.sha256(original_spot_bytes).hexdigest(),
        quality=quality,
        forensic=forensic,
        plan=plan,
        plan_path=plan_path,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        contract_root=Path(contract["root"]),
    )
    original_folds_path, original_folds_bytes = _read_output(
        original_spot,
        key="rolling_spot_profile_folds_csv",
        label="legacy rolling folds CSV",
    )
    forensic_folds_path, forensic_folds_bytes = _read_output(
        forensic,
        key="rolling_spot_profile_folds_csv",
        label="forensic rolling folds CSV",
    )
    forensic_post_path, forensic_post_bytes = _read_output(
        forensic,
        key="post_valuation_timestamp_residuals_csv",
        label="forensic post-valuation CSV",
    )
    original_integrity = _fold_integrity(original_folds_bytes, label="legacy rolling folds CSV")
    corrected_integrity = _fold_integrity(forensic_folds_bytes, label="forensic rolling folds CSV")
    if not (
        original_integrity["row_count"] == 12
        and original_integrity["unique_cutoff_count"] == 1
        and original_integrity["duplicate_cutoff_count"] == 11
        and original_integrity["non_overlapping_evaluations"] is False
    ):
        raise ValueError("legacy T057 fold defect was not reproduced exactly")
    if not (
        corrected_integrity["row_count"] == 1
        and corrected_integrity["unique_cutoff_count"] == 1
        and corrected_integrity["duplicate_cutoff_count"] == 0
        and corrected_integrity["non_overlapping_evaluations"] is True
    ):
        raise ValueError("forensic replay does not establish one corrected fold")

    recomputed_rolling_metrics = _rolling_metrics(
        forensic_folds_bytes,
        label="forensic rolling folds CSV",
    )
    if not _metrics_match(forensic.get("rolling_metrics"), recomputed_rolling_metrics):
        raise ValueError("forensic rolling metrics do not match the fold CSV")
    recomputed_future_metrics = _post_metrics(
        forensic_post_bytes,
        label="forensic post-valuation CSV",
        expected_start=str(plan.get("holdout_start_utc", "")),
        expected_end=str(plan.get("holdout_end_utc", "")),
    )
    if not _metrics_match(forensic.get("post_valuation_metrics"), recomputed_future_metrics):
        raise ValueError("forensic future metrics do not match the post-valuation CSV")
    future_metrics = _mapping(locked.get("holdout_metrics"), label="future holdout metrics")
    forensic_future_metrics = _mapping(
        forensic.get("post_valuation_metrics"),
        label="forensic future holdout metrics",
    )
    if not _metrics_match(future_metrics, forensic_future_metrics) or not _metrics_match(
        future_metrics,
        recomputed_future_metrics,
    ):
        raise ValueError("forensic replay changed the future holdout metrics")
    if int(future_metrics.get("hours", 0)) != 336:
        raise ValueError("future T057 holdout must contain exactly 336 hours")

    correction: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "T057_HISTORICAL_ROLLING_EVIDENCE_REVOKED",
        "production_authorization": False,
        "promotion_gate": False,
        "plan_id": "t057_locked_t056_future_holdout",
        "source_contract": {
            "canonical_plan": str(plan_path),
            "canonical_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "canonical_sources_only": True,
            "caller_anchors_match_canonical_registry": True,
            "candidate_spot_and_config_cross_bound": True,
            "metrics_recomputed_from_csv": True,
        },
        "supersession": {
            "superseded_quality_report": str(quality_path),
            "superseded_quality_report_sha256": hashlib.sha256(quality_bytes).hexdigest(),
            "superseded_status": quality.get("status"),
            "withdrawn_claim": "12/12 rolling-origin folds improve MAE",
            "replacement_claim": "one distinct historical fold; historical robustness not established",
        },
        "legacy_evidence": {
            "run_summary": str(original_path),
            "run_summary_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "spot_backtest_summary": str(original_spot_path),
            "spot_backtest_summary_sha256": hashlib.sha256(original_spot_bytes).hexdigest(),
            "rolling_folds_csv": str(original_folds_path),
            "rolling_folds_csv_sha256": hashlib.sha256(original_folds_bytes).hexdigest(),
            "fold_integrity": original_integrity,
        },
        "forensic_replay": {
            "promotable": False,
            "backtest_summary": str(forensic_path),
            "backtest_summary_sha256": hashlib.sha256(forensic_bytes).hexdigest(),
            "rolling_folds_csv": str(forensic_folds_path),
            "rolling_folds_csv_sha256": hashlib.sha256(forensic_folds_bytes).hexdigest(),
            "post_valuation_csv": str(forensic_post_path),
            "post_valuation_csv_sha256": hashlib.sha256(forensic_post_bytes).hexdigest(),
            "fold_integrity": corrected_integrity,
            "rolling_metrics": recomputed_rolling_metrics,
        },
        "retained_future_holdout": {
            "scientific_status": "PARTIAL_POSITIVE_FUTURE_HOLDOUT_SINGLE_EPISODE",
            "metrics": recomputed_future_metrics,
            "independent_episode_count": 1,
            "historical_robustness_established": False,
            "materiality_established": False,
        },
        "decision": {
            "legacy_locked_holdout_pass_accepted": False,
            "legacy_quality_report_accepted": False,
            "production_go": False,
            "rerun_legacy_one_shot_directory": False,
            "required_next_evidence": (
                "new immutable plan with preregistered unique PIT origins, full per-origin refit, "
                "multiple future regimes, and probabilistic calibration"
            ),
        },
    }
    output_path = _output_path(output)
    payload = _canonical_json(correction)
    write_durable_exact_bytes(output_path, payload, label="T057 fold-integrity correction")
    return correction


def _canonical_source_contract() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "original_run": root / _ORIGINAL_RUN_RELATIVE_PATH,
        "original_run_sha256": _ORIGINAL_RUN_SHA256,
        "superseded_quality": root / _SUPERSEDED_QUALITY_RELATIVE_PATH,
        "superseded_quality_sha256": _SUPERSEDED_QUALITY_SHA256,
        "forensic_summary": root / _FORENSIC_SUMMARY_RELATIVE_PATH,
        "forensic_summary_sha256": _FORENSIC_SUMMARY_SHA256,
        "plan": canonical_t057_plan_path(),
        "plan_sha256": T057_PLAN_SHA256,
    }


def _require_contract_source(
    actual_path: Path,
    caller_sha256: str,
    *,
    expected_path: Any,
    expected_sha256: str,
    label: str,
) -> None:
    if _input_path(actual_path) != _input_path(Path(expected_path)):
        raise ValueError(f"{label} is not the canonical registered source")
    if _sha256_value(caller_sha256, label=label) != _sha256_value(
        expected_sha256,
        label=f"registered {label}",
    ):
        raise ValueError(f"{label} caller anchor is not the canonical registered SHA-256")


def _validate_canonical_cross_bindings(
    *,
    original: Mapping[str, Any],
    original_sha256: str,
    original_spot: Mapping[str, Any],
    original_spot_sha256: str,
    quality: Mapping[str, Any],
    forensic: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_sha256: str,
    contract_root: Path,
) -> None:
    if plan.get("plan_id") != T057_PLAN_ID or plan_sha256 != str(
        _canonical_source_contract()["plan_sha256"]
    ):
        raise ValueError("canonical T057 plan identity is invalid")
    locked = _mapping(original.get("locked_holdout"), label="legacy locked holdout")
    identity = _mapping(locked.get("locked_plan_identity"), label="legacy locked plan identity")
    if not (
        original.get("actual_plan_json_sha256") == plan_sha256
        and original.get("expected_plan_json_sha256") == plan_sha256
        and _evidence_path(original.get("plan_json"), root=contract_root) == plan_path
        and identity.get("plan_id") == T057_PLAN_ID
        and identity.get("plan_json_sha256") == plan_sha256
        and _evidence_path(identity.get("plan_json"), root=contract_root) == plan_path
        and locked.get("actual_plan_json_sha256") == plan_sha256
        and locked.get("expected_plan_json_sha256") == plan_sha256
    ):
        raise ValueError("legacy wrapper is not cross-bound to the canonical T057 plan")

    quality_source = _mapping(quality.get("source"), label="quality source")
    quality_evidence = _mapping(quality_source.get("evidence"), label="quality evidence")
    if not (
        quality_source.get("run_id") == T057_PLAN_ID
        and quality_source.get("plan_sha256") == plan_sha256
        and quality_source.get("run_summary_sha256") == original_sha256
        and _evidence_sha(quality_evidence, "spot backtest summary") == original_spot_sha256
        and _evidence_sha(quality_evidence, "locked holdout run summary")
        == original.get("locked_holdout_run_summary_sha256")
        and _evidence_sha(quality_evidence, "locked plan") == plan_sha256
    ):
        raise ValueError("superseded quality report is not cross-bound to the canonical T057 run")

    forensic_hashes = _mapping(forensic.get("source_hashes"), label="forensic source hashes")
    legacy_hashes = _mapping(original_spot.get("source_hashes"), label="legacy source hashes")
    for key in ("baseline_csv", "adjusted_csv"):
        source_path = _evidence_path(plan.get(key), root=contract_root)
        source_sha = str(plan.get(f"{key}_sha256", ""))
        source_bytes = read_stable_single_link_file(
            source_path,
            label=f"canonical T057 {key}",
            max_bytes=_MAX_CSV_BYTES,
        )
        if not (
            hashlib.sha256(source_bytes).hexdigest() == source_sha
            and identity.get(key) == plan.get(key)
            and identity.get(f"{key}_sha256") == source_sha
            and _evidence_path(forensic.get(key), root=contract_root) == source_path
            and forensic_hashes.get(key) == source_sha
            and legacy_hashes.get(key) == source_sha
            and _evidence_sha(
                quality_evidence,
                "baseline candidate" if key == "baseline_csv" else "adjusted candidate",
            )
            == source_sha
        ):
            raise ValueError(f"canonical T057 {key} provenance is inconsistent")

    spot_path = _evidence_path(locked.get("spot_parquet"), root=contract_root)
    spot_sha = str(_mapping(locked.get("spot_provenance"), label="spot provenance").get(
        "spot_parquet_sha256",
        "",
    ))
    spot_bytes = read_stable_single_link_file(
        spot_path,
        label="canonical T057 spot parquet",
        max_bytes=_MAX_CSV_BYTES,
    )
    if not (
        hashlib.sha256(spot_bytes).hexdigest() == spot_sha
        and _evidence_path(forensic.get("spot_parquet"), root=contract_root) == spot_path
        and forensic_hashes.get("spot_parquet") == spot_sha
        and legacy_hashes.get("spot_parquet") == spot_sha
        and _evidence_sha(quality_evidence, "spot parquet") == spot_sha
    ):
        raise ValueError("canonical T057 spot provenance is inconsistent")

    cfg = _mapping(plan.get("backtest"), label="canonical backtest configuration")
    evaluation = _mapping(forensic.get("evaluation_window"), label="forensic evaluation window")
    if not (
        _utc_text(_utc(str(forensic.get("valuation_timestamp_utc")), label="valuation"))
        == _utc_text(_utc(str(cfg.get("valuation_timestamp_utc")), label="plan valuation"))
        and int(forensic.get("lookback_years", -1)) == int(cfg.get("lookback_years", -2))
        and int(forensic.get("eval_days", -1)) == int(cfg.get("eval_days", -2))
        and int(forensic.get("embargo_days", -1)) == int(cfg.get("embargo_days", -2))
        and _utc_text(_utc(str(evaluation.get("start_utc")), label="evaluation start"))
        == _utc_text(_utc(str(plan.get("holdout_start_utc")), label="plan holdout start"))
        and _utc_text(_utc(str(evaluation.get("end_utc")), label="evaluation end"))
        == _utc_text(_utc(str(plan.get("holdout_end_utc")), label="plan holdout end"))
        and evaluation.get("end_exclusive") is True
        and int(forensic.get("rolling_fold_count", -1)) == 1
        and int(forensic.get("eligible_rolling_fold_count", -1)) == 1
    ):
        raise ValueError("forensic replay configuration is not the canonical T057 replay")


def _evidence_sha(evidence: Mapping[str, Any], key: str) -> str:
    return str(_mapping(evidence.get(key), label=f"quality evidence {key}").get("sha256", ""))


def _evidence_path(value: Any, *, root: Path) -> Path:
    selected = Path(str(value)).expanduser()
    if not selected.is_absolute():
        selected = root / selected
    return assert_absolute_path_has_no_links(Path(selected.absolute()))


def _read_anchored(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> tuple[Path, bytes]:
    selected = _input_path(path)
    payload = read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)
    if hashlib.sha256(payload).hexdigest() != _sha256_value(expected_sha256, label=label):
        raise ValueError(f"{label} does not match the caller-held SHA-256")
    return selected, payload


def _read_linked(
    document: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    label: str,
    max_bytes: int,
) -> tuple[Path, bytes]:
    return _read_anchored(
        Path(str(document.get(path_key, ""))),
        str(document.get(sha_key, "")),
        label=label,
        max_bytes=max_bytes,
    )


def _read_output(
    summary: Mapping[str, Any],
    *,
    key: str,
    label: str,
) -> tuple[Path, bytes]:
    outputs = _mapping(summary.get("outputs"), label="backtest outputs")
    hashes = _mapping(summary.get("output_hashes"), label="backtest output hashes")
    return _read_anchored(
        _evidence_path(outputs.get(key), root=Path(_canonical_source_contract()["root"])),
        str(hashes.get(key, "")),
        label=label,
        max_bytes=_MAX_CSV_BYTES,
    )


def _fold_integrity(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {"cutoff_utc", "eval_start_utc", "eval_end_exclusive_utc"}
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(set(reader.fieldnames))
        or not required.issubset(reader.fieldnames)
    ):
        raise ValueError(f"{label} schema is invalid")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{label} is empty")
    cutoffs = [_utc(row["cutoff_utc"], label=label) for row in rows]
    starts = [_utc(row["eval_start_utc"], label=label) for row in rows]
    ends = [_utc(row["eval_end_exclusive_utc"], label=label) for row in rows]
    return {
        "row_count": len(rows),
        "unique_cutoff_count": len(set(cutoffs)),
        "duplicate_cutoff_count": len(cutoffs) - len(set(cutoffs)),
        "cutoffs_strictly_ordered": cutoffs == sorted(set(cutoffs)),
        "non_overlapping_evaluations": bool(
            all(start < end for start, end in zip(starts, ends, strict=True))
            and all(left <= right for left, right in zip(ends, starts[1:]))
        ),
        "first_cutoff_utc": _utc_text(min(cutoffs)),
        "last_cutoff_utc": _utc_text(max(cutoffs)),
    }


def _rolling_metrics(payload: bytes, *, label: str) -> dict[str, Any]:
    rows = _csv_rows(payload, label=label)
    required = {
        "fold_eligible",
        "baseline_profile_mae_eur_mwh",
        "adjusted_profile_mae_eur_mwh",
        "profile_mae_improvement_eur_mwh",
        "baseline_profile_correlation",
        "adjusted_profile_correlation",
        "train_eval_profile_mae_eur_mwh",
        "historical_fold_not_independent_of_current_candidate_fit",
        "eval_after_current_valuation",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"{label} is missing rolling metric columns")
    eligible = [row for row in rows if _bool(row["fold_eligible"], label=label)]
    if len(eligible) != 1:
        raise ValueError(f"{label} must establish exactly one eligible corrected fold")
    improvements = [_number(row["profile_mae_improvement_eur_mwh"], label=label) for row in eligible]
    return {
        "mean_baseline_profile_mae_eur_mwh": _mean_field(
            eligible,
            "baseline_profile_mae_eur_mwh",
            label=label,
        ),
        "mean_adjusted_profile_mae_eur_mwh": _mean_field(
            eligible,
            "adjusted_profile_mae_eur_mwh",
            label=label,
        ),
        "mean_profile_mae_improvement_eur_mwh": math.fsum(improvements) / len(improvements),
        "median_profile_mae_improvement_eur_mwh": sorted(improvements)[len(improvements) // 2],
        "positive_mae_improvement_fold_count": sum(value > 0.0 for value in improvements),
        "mean_baseline_profile_correlation": _mean_field(
            eligible,
            "baseline_profile_correlation",
            label=label,
        ),
        "mean_adjusted_profile_correlation": _mean_field(
            eligible,
            "adjusted_profile_correlation",
            label=label,
        ),
        "mean_train_eval_profile_mae_eur_mwh": _mean_field(
            eligible,
            "train_eval_profile_mae_eur_mwh",
            label=label,
        ),
        "historical_fold_not_independent_count": sum(
            _bool(
                row["historical_fold_not_independent_of_current_candidate_fit"],
                label=label,
            )
            for row in eligible
        ),
        "post_valuation_fold_count": sum(
            _bool(row["eval_after_current_valuation"], label=label) for row in eligible
        ),
    }


def _post_metrics(
    payload: bytes,
    *,
    label: str,
    expected_start: str,
    expected_end: str,
) -> dict[str, Any]:
    rows = _csv_rows(payload, label=label)
    required = {
        "timestamp_utc",
        "baseline_abs_error_eur_mwh",
        "adjusted_abs_error_eur_mwh",
    }
    if not required.issubset(rows[0]):
        raise ValueError(f"{label} is missing holdout metric columns")
    timestamps = [_utc(row["timestamp_utc"], label=label) for row in rows]
    start = _utc(expected_start, label="expected T057 holdout start")
    end = _utc(expected_end, label="expected T057 holdout end")
    expected_timestamps = [start + timedelta(hours=index) for index in range(336)]
    if timestamps != expected_timestamps or expected_timestamps[-1] >= end:
        raise ValueError(f"{label} does not contain the exact 336-hour T057 window")
    baseline = [_number(row["baseline_abs_error_eur_mwh"], label=label) for row in rows]
    adjusted = [_number(row["adjusted_abs_error_eur_mwh"], label=label) for row in rows]
    baseline_mean = math.fsum(baseline) / len(baseline)
    adjusted_mean = math.fsum(adjusted) / len(adjusted)
    return {
        "hours": len(rows),
        "baseline_residual_mae_eur_mwh": baseline_mean,
        "adjusted_residual_mae_eur_mwh": adjusted_mean,
        "residual_mae_improvement_eur_mwh": baseline_mean - adjusted_mean,
    }


def _csv_rows(payload: bytes, *, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if (
        reader.fieldnames is None
        or len(reader.fieldnames) != len(set(reader.fieldnames))
        or any(not field for field in reader.fieldnames)
    ):
        raise ValueError(f"{label} CSV header is invalid")
    rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError(f"{label} CSV rows are invalid")
    return rows


def _mean_field(rows: list[dict[str, str]], field: str, *, label: str) -> float:
    values = [_number(row[field], label=label) for row in rows]
    return math.fsum(values) / len(values)


def _number(value: str, *, label: str) -> float:
    try:
        selected = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains an invalid number") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} contains a non-finite number")
    return selected


def _bool(value: str, *, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{label} contains an invalid boolean")
    return normalized == "true"


def _metrics_match(actual: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, int):
            if isinstance(actual_value, bool) or actual_value != expected_value:
                return False
        else:
            try:
                if not math.isfinite(float(actual_value)) or not math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    def unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key}")
            result[key] = item
        return result

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


def _input_path(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return assert_absolute_path_has_no_links(Path(selected.absolute()))


def _output_path(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    selected = Path(selected.absolute())
    parent = assert_absolute_path_has_no_links(selected.parent)
    parent.mkdir(parents=True, exist_ok=True)
    admitted = assert_absolute_path_has_no_links(parent / selected.name)
    if admitted.exists():
        raise FileExistsError(f"T057 correction output is append-only: {admitted}")
    return admitted


def _sha256_value(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} SHA-256 is invalid")
    return value


def _utc(value: str, *, label: str) -> datetime:
    try:
        selected = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is invalid") from exc
    if selected.tzinfo is None:
        raise ValueError(f"{label} timestamp must carry a UTC offset")
    return selected.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run-summary", type=Path, required=True)
    parser.add_argument("--expected-original-run-summary-sha256", required=True)
    parser.add_argument("--superseded-quality-report", type=Path, required=True)
    parser.add_argument("--expected-superseded-quality-report-sha256", required=True)
    parser.add_argument("--forensic-backtest-summary", type=Path, required=True)
    parser.add_argument("--expected-forensic-backtest-summary-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = build_correction(
        original_run_summary=args.original_run_summary,
        expected_original_run_summary_sha256=args.expected_original_run_summary_sha256,
        superseded_quality_report=args.superseded_quality_report,
        expected_superseded_quality_report_sha256=args.expected_superseded_quality_report_sha256,
        forensic_backtest_summary=args.forensic_backtest_summary,
        expected_forensic_backtest_summary_sha256=args.expected_forensic_backtest_summary_sha256,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
