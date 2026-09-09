"""Build a deterministic, non-promotional local quality view from locked T057 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping

import scripts.epex_lab_locked_holdout_policy as holdout_policy
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from scripts.verify_t057_evidence_supersession import supersession_record_for_sha256

REPORT_SCHEMA = "pfc_local_quality_report.v2"
RUN_SCHEMA = holdout_policy.ENERGY_CHARTS_RUN_SCHEMA_VERSION
_SHA256_LENGTH = 64
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_BOUND_ARTIFACT_BYTES = 256 * 1024 * 1024


def build_local_quality_report(
    *,
    run_summary: str | Path,
    expected_run_summary_sha256: str,
    repo_root: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, object]:
    """Verify locked evidence and emit a stable local quality report."""

    repo = assert_absolute_path_has_no_links(Path(repo_root))
    if not repo.is_dir():
        raise ValueError("local quality repo root must be a directory")
    summary_path = assert_absolute_path_has_no_links(Path(run_summary))
    run_root = summary_path.parent
    json_output = _admit_output(Path(output_json), run_root=run_root)
    markdown_output = _admit_output(Path(output_markdown), run_root=run_root)
    if json_output == markdown_output:
        raise ValueError("local quality outputs must be distinct")

    top_payload = _read_stable(summary_path, max_bytes=_MAX_JSON_BYTES)
    expected_top_sha256 = _sha256_value(
        expected_run_summary_sha256,
        label="expected T057 run summary SHA-256",
    )
    actual_top_sha256 = hashlib.sha256(top_payload).hexdigest()
    if actual_top_sha256 != expected_top_sha256:
        raise ValueError("T057 run summary SHA-256 does not match the caller-held anchor")
    supersession = supersession_record_for_sha256(actual_top_sha256)
    if supersession is not None:
        authority = _mapping(
            supersession.get("authoritative_correction"),
            label="T057 supersession authority",
        )
        raise ValueError(
            "T057 run summary is superseded fail-closed by "
            f"{authority.get('path')} ({authority.get('sha256')})"
        )
    top = _strict_mapping(top_payload, label="T057 run summary")
    if top.get("schema_version") != RUN_SCHEMA:
        raise ValueError("local quality input run schema is unsupported")
    if top.get("production_approved") is not False or top.get("promotion_gate") is not False:
        raise ValueError("local quality input must remain non-promotional")
    if any(top.get(key) is not False for key in _ompex_keys()):
        raise ValueError("local quality input must not use OMPEX")
    _require_canonical_t057_source(summary_path=summary_path, top=top, repo_root=repo)

    bound: dict[str, dict[str, object]] = {}
    locked_summary = _read_bound_json(
        top,
        path_key="locked_holdout_run_summary",
        sha_key="locked_holdout_run_summary_sha256",
        run_root=run_root,
        label="locked holdout run summary",
        bound=bound,
    )
    locked = _mapping(top.get("locked_holdout"), label="locked holdout result")
    if locked_summary != locked:
        raise ValueError("linked locked holdout summary does not match embedded result")
    audit = _read_bound_json(
        locked,
        path_key="locked_holdout_audit",
        sha_key="locked_holdout_audit_sha256",
        run_root=run_root,
        label="locked holdout audit",
        bound=bound,
    )
    spot = _read_bound_json(
        locked,
        path_key="spot_backtest_summary",
        sha_key="spot_backtest_summary_sha256",
        run_root=run_root,
        label="spot backtest summary",
        bound=bound,
    )
    _require_audit_backtest_binding(audit=audit, locked=locked)
    _read_bound_json(
        locked,
        path_key="coverage_status",
        sha_key="coverage_status_sha256",
        run_root=run_root,
        label="coverage status",
        bound=bound,
    )
    for path_key, sha_key, label in (
        ("attempt_seal", "attempt_seal_sha256", "attempt seal"),
        ("capture_seal", "capture_seal_sha256", "capture seal"),
        ("provider_raw_json", "provider_raw_json_sha256", "provider raw"),
        ("spot_fetch_summary", "spot_fetch_summary_sha256", "spot fetch summary"),
    ):
        _read_bound_bytes(
            top,
            path_key=path_key,
            sha_key=sha_key,
            allowed_root=run_root,
            label=label,
            bound=bound,
        )

    folds_payload = _read_spot_output(
        spot,
        output_key="rolling_spot_profile_folds_csv",
        run_root=run_root,
        label="rolling folds CSV",
        bound=bound,
    )
    buckets_payload = _read_spot_output(
        spot,
        output_key="rolling_spot_bucket_metrics_csv",
        run_root=run_root,
        label="rolling bucket CSV",
        bound=bound,
    )

    spot_fetch = _mapping(top.get("spot_fetch"), label="spot fetch result")
    _read_bound_bytes(
        spot_fetch,
        path_key="output_parquet",
        sha_key="output_parquet_sha256",
        allowed_root=run_root,
        label="spot parquet",
        bound=bound,
    )

    plan_identity = _mapping(top.get("locked_plan_identity"), label="locked plan identity")
    for path_key, sha_key, label in (
        ("plan_json", "plan_json_sha256", "locked plan"),
        ("baseline_csv", "baseline_csv_sha256", "baseline candidate"),
        ("adjusted_csv", "adjusted_csv_sha256", "adjusted candidate"),
        ("lab_manifest", "lab_manifest_sha256", "lab manifest"),
        ("selection_summary", "selection_summary_sha256", "selection summary"),
    ):
        _read_repo_bound_bytes(
            plan_identity,
            path_key=path_key,
            sha_key=sha_key,
            repo_root=repo,
            label=label,
            bound=bound,
        )

    policy = holdout_policy.locked_holdout_policy(top)
    if policy.get("pass") is not True:
        failed = sorted(
            key
            for key, value in _mapping(
                policy.get("checks"), label="locked holdout policy checks"
            ).items()
            if value is not True
        )
        raise ValueError(f"locked T057 holdout policy did not pass: {failed}")
    _require_audit_fold_replay(audit)

    holdout_metrics = _metric_mapping(
        locked.get("holdout_metrics"),
        label="holdout metrics",
    )
    audit_metrics = _metric_mapping(audit.get("holdout_metrics"), label="audit metrics")
    if holdout_metrics != audit_metrics:
        raise ValueError("local quality holdout metrics do not match independent audit")
    locked_metrics = _metric_mapping(
        locked_summary.get("holdout_metrics"),
        label="locked summary metrics",
    )
    if holdout_metrics != locked_metrics:
        raise ValueError("local quality holdout metrics do not match locked summary")

    rolling, rolling_fold_count = _recompute_rolling_metrics(folds_payload)
    rolling_buckets = _recompute_bucket_metrics(
        buckets_payload,
        eligible_cutoffs=rolling.pop("_eligible_cutoffs"),
    )
    _require_declared_aggregate_matches(
        declared=spot.get("rolling_metrics"),
        recomputed=rolling,
        label="rolling metrics",
    )
    _require_declared_aggregate_matches(
        declared=spot.get("rolling_bucket_metrics"),
        recomputed=rolling_buckets,
        label="rolling bucket metrics",
    )
    declared_fold_count = int(
        _finite_number(spot.get("rolling_fold_count"), label="rolling fold count")
    )
    declared_eligible_fold_count = int(
        _finite_number(
            spot.get("eligible_rolling_fold_count"),
            label="eligible rolling fold count",
        )
    )
    if (
        declared_fold_count != rolling_fold_count
        or declared_eligible_fold_count != rolling_fold_count
    ):
        raise ValueError("rolling fold counts do not match the captured folds CSV")
    buckets = _bucket_metrics(rolling_buckets)
    gates = _quality_gates(top=top, locked=locked, audit=audit, spot=spot)
    gates["locked_t057_policy_pass"] = True
    gates["rolling_aggregates_recomputed_from_captured_csv"] = True
    if not all(gates.values()):
        failed = sorted(key for key, value in gates.items() if not value)
        raise ValueError(f"local quality source gates did not pass: {failed}")

    baseline_mae = holdout_metrics["baseline_residual_mae_eur_mwh"]
    adjusted_mae = holdout_metrics["adjusted_residual_mae_eur_mwh"]
    holdout_uplift = holdout_metrics["residual_mae_improvement_eur_mwh"]
    rolling_baseline = _finite_number(
        rolling.get("mean_baseline_profile_mae_eur_mwh"),
        label="rolling baseline profile MAE",
    )
    rolling_adjusted = _finite_number(
        rolling.get("mean_adjusted_profile_mae_eur_mwh"),
        label="rolling adjusted profile MAE",
    )
    rolling_uplift = _finite_number(
        rolling.get("mean_profile_mae_improvement_eur_mwh"),
        label="rolling profile MAE improvement",
    )
    positive_rolling_folds = int(
        _finite_number(
            rolling.get("positive_mae_improvement_fold_count"),
            label="positive rolling folds",
        )
    )
    _require_positive_uplift(
        baseline_mae,
        adjusted_mae,
        holdout_uplift,
        label="future holdout",
    )
    _require_positive_uplift(
        rolling_baseline,
        rolling_adjusted,
        rolling_uplift,
        label="rolling-origin",
    )
    if rolling_fold_count < 1 or positive_rolling_folds != rolling_fold_count:
        raise ValueError("rolling-origin positive fold count must equal the fold count")
    for name, bucket in buckets.items():
        _require_positive_uplift(
            float(bucket["baseline_mae_eur_mwh"]),
            float(bucket["adjusted_mae_eur_mwh"]),
            float(bucket["mae_improvement_eur_mwh"]),
            label=f"rolling bucket {name}",
        )
        if int(bucket["fold_count"]) < 1 or int(bucket["positive_improvement_fold_count"]) != int(
            bucket["fold_count"]
        ):
            raise ValueError(f"rolling bucket {name} positive fold count is incomplete")

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "status": "LOCAL_DIAGNOSTIC_PASS_NOT_PRODUCTION",
        "production_authorization": False,
        "promotion_gate": False,
        "source": {
            "run_id": str(plan_identity.get("plan_id", "")),
            "run_summary_sha256": actual_top_sha256,
            "plan_sha256": str(plan_identity.get("plan_json_sha256", "")),
            "evidence": {key: bound[key] for key in sorted(bound)},
        },
        "coverage": {
            "holdout_start_utc": str(top.get("holdout_start_utc", "")),
            "holdout_end_utc": str(top.get("holdout_end_utc", "")),
            "holdout_hours": int(
                _finite_number(
                    _mapping(locked.get("coverage"), label="coverage").get(
                        "expected_holdout_hours"
                    ),
                    label="holdout hours",
                )
            ),
            "missing_holdout_hours": int(
                _finite_number(
                    _mapping(locked.get("coverage"), label="coverage").get("missing_holdout_hours"),
                    label="missing holdout hours",
                )
            ),
            "provider_history_hours": int(
                _finite_number(
                    spot_fetch.get("observed_hour_count"),
                    label="provider history hours",
                )
            ),
        },
        "holdout": {
            "baseline_mae_eur_mwh": baseline_mae,
            "adjusted_mae_eur_mwh": adjusted_mae,
            "mae_improvement_eur_mwh": holdout_uplift,
            "relative_mae_improvement_percent": _percentage(holdout_uplift, baseline_mae),
        },
        "rolling_origin": {
            "fold_count": rolling_fold_count,
            "positive_improvement_fold_count": positive_rolling_folds,
            "baseline_profile_mae_eur_mwh": rolling_baseline,
            "adjusted_profile_mae_eur_mwh": rolling_adjusted,
            "mae_improvement_eur_mwh": rolling_uplift,
            "relative_mae_improvement_percent": _percentage(
                rolling_uplift,
                rolling_baseline,
            ),
            "historical_non_independent_fold_count": int(
                _finite_number(
                    rolling.get("historical_fold_not_independent_count"),
                    label="historical non-independent folds",
                )
            ),
            "post_valuation_fold_count": int(
                _finite_number(
                    rolling.get("post_valuation_fold_count"),
                    label="post-valuation folds",
                )
            ),
        },
        "bucket_metrics": buckets,
        "gates": gates,
        "assessment": {
            "scientific_status": "POSITIVE_LOCKED_HOLDOUT_UPLIFT",
            "materiality_status": "NOT_ESTABLISHED_BY_T057_PLAN",
            "main_strength": (
                "tous les folds rolling-origin enregistrés et le holdout futur améliorent la MAE"
            ),
            "main_limit": (
                "T057 couvre 336 heures futures ; les folds historiques ne constituent "
                "pas une preuve de production indépendante"
            ),
        },
        "limitations": [
            "Ce rapport est un diagnostic local et ne peut pas autoriser la production.",
            "Le plan verrouillé teste un uplift positif, pas sa matérialité économique.",
            "Le holdout futur couvre 336 heures et doit être complété par d'autres holdouts.",
            "Les folds historiques sont informatifs mais pas des preuves futures indépendantes.",
            "CAS externe, identités de service, ACL, Docker et drills restent non prouvés.",
        ],
    }
    json_payload = _canonical_json(report)
    markdown_payload = _render_markdown(report).encode("utf-8")
    write_durable_exact_bytes(json_output, json_payload, label="local PFC quality JSON")
    write_durable_exact_bytes(
        markdown_output,
        markdown_payload,
        label="local PFC quality Markdown",
    )
    return report


def _require_canonical_t057_source(
    *,
    summary_path: Path,
    top: Mapping[str, object],
    repo_root: Path,
) -> None:
    canonical_summary = (
        holdout_policy.canonical_t057_output_path()
        / "energy_charts_locked_holdout_run_summary.json"
    )
    if summary_path != canonical_summary:
        raise ValueError("local quality input must use the canonical T057 run summary route")
    identity = _mapping(top.get("locked_plan_identity"), label="locked plan identity")
    if identity.get("plan_id") != holdout_policy.T057_PLAN_ID:
        raise ValueError("local quality input plan id is not canonical T057")
    if identity.get("plan_json_sha256") != holdout_policy.T057_PLAN_SHA256:
        raise ValueError("local quality input plan SHA-256 is not canonical T057")
    raw_plan = Path(str(identity.get("plan_json", "")))
    selected_plan = raw_plan if raw_plan.is_absolute() else repo_root / raw_plan
    selected_plan = assert_absolute_path_has_no_links(selected_plan)
    if selected_plan != holdout_policy.canonical_t057_plan_path():
        raise ValueError("local quality input plan path is not canonical T057")


def _require_audit_backtest_binding(
    *,
    audit: Mapping[str, object],
    locked: Mapping[str, object],
) -> None:
    if audit.get("spot_backtest_summary") != locked.get("spot_backtest_summary"):
        raise ValueError("locked audit backtest path does not match locked run")
    if audit.get("spot_backtest_summary_sha256") != locked.get("spot_backtest_summary_sha256"):
        raise ValueError("locked audit backtest SHA-256 does not match locked run")


def _require_audit_fold_replay(audit: Mapping[str, object]) -> None:
    checks = _mapping(audit.get("checks"), label="audit checks")
    required = (
        "rolling_folds_temporal_integrity_independently_replayed",
        "rolling_folds_unique_ordered_cutoffs_independently_replayed",
        "rolling_folds_non_overlapping_evaluations_independently_replayed",
    )
    failed = [name for name in required if checks.get(name) is not True]
    if failed:
        raise ValueError(f"locked audit did not independently replay fold integrity: {failed}")


def _read_spot_output(
    spot: Mapping[str, object],
    *,
    output_key: str,
    run_root: Path,
    label: str,
    bound: dict[str, dict[str, object]],
) -> bytes:
    outputs = _mapping(spot.get("outputs"), label="spot backtest outputs")
    hashes = _mapping(spot.get("output_hashes"), label="spot backtest output hashes")
    return _read_bound_bytes(
        {"path": outputs.get(output_key), "sha256": hashes.get(output_key)},
        path_key="path",
        sha_key="sha256",
        allowed_root=run_root,
        label=label,
        bound=bound,
    )


def _recompute_rolling_metrics(payload: bytes) -> tuple[dict[str, object], int]:
    rows = _csv_rows(
        payload,
        label="rolling folds CSV",
        required_columns={
            "cutoff_utc",
            "eval_start_utc",
            "eval_end_exclusive_utc",
            "fold_eligible",
            "baseline_profile_mae_eur_mwh",
            "adjusted_profile_mae_eur_mwh",
            "profile_mae_improvement_eur_mwh",
            "baseline_profile_correlation",
            "adjusted_profile_correlation",
            "train_eval_profile_mae_eur_mwh",
            "historical_fold_not_independent_of_current_candidate_fit",
            "eval_after_current_valuation",
        },
    )
    if not rows:
        raise ValueError("rolling folds CSV must contain at least one fold")
    cutoffs = [_utc_datetime(row["cutoff_utc"], label="rolling cutoff") for row in rows]
    eval_starts = [_utc_datetime(row["eval_start_utc"], label="rolling eval start") for row in rows]
    eval_ends = [
        _utc_datetime(row["eval_end_exclusive_utc"], label="rolling eval end") for row in rows
    ]
    if len(set(cutoffs)) != len(cutoffs) or cutoffs != sorted(cutoffs):
        raise ValueError("captured rolling cutoffs must be unique and strictly ordered")
    if any(start >= end for start, end in zip(eval_starts, eval_ends, strict=True)):
        raise ValueError("captured rolling evaluation window is invalid")
    if any(left_end > right_start for left_end, right_start in zip(eval_ends, eval_starts[1:])):
        raise ValueError("captured rolling evaluation windows overlap")
    eligible = [row for row in rows if _csv_bool(row["fold_eligible"], label="fold eligible")]
    if len(eligible) != len(rows):
        raise ValueError("all captured rolling folds must be eligible for the quality report")

    def values(column: str) -> list[float]:
        return [_csv_float(row[column], label=f"rolling folds {column}") for row in eligible]

    improvements = values("profile_mae_improvement_eur_mwh")
    result: dict[str, object] = {
        "mean_baseline_profile_mae_eur_mwh": _mean_values(values("baseline_profile_mae_eur_mwh")),
        "mean_adjusted_profile_mae_eur_mwh": _mean_values(values("adjusted_profile_mae_eur_mwh")),
        "mean_profile_mae_improvement_eur_mwh": _mean_values(improvements),
        "median_profile_mae_improvement_eur_mwh": float(statistics.median(improvements)),
        "positive_mae_improvement_fold_count": sum(value > 0.0 for value in improvements),
        "mean_baseline_profile_correlation": _mean_values(values("baseline_profile_correlation")),
        "mean_adjusted_profile_correlation": _mean_values(values("adjusted_profile_correlation")),
        "mean_train_eval_profile_mae_eur_mwh": _mean_values(
            values("train_eval_profile_mae_eur_mwh")
        ),
        "historical_fold_not_independent_count": sum(
            _csv_bool(
                row["historical_fold_not_independent_of_current_candidate_fit"],
                label="historical fold independence",
            )
            for row in eligible
        ),
        "post_valuation_fold_count": sum(
            _csv_bool(row["eval_after_current_valuation"], label="post valuation fold")
            for row in eligible
        ),
        "_eligible_cutoffs": {row["cutoff_utc"] for row in eligible},
    }
    return result, len(rows)


def _recompute_bucket_metrics(
    payload: bytes,
    *,
    eligible_cutoffs: object,
) -> dict[str, dict[str, object]]:
    rows = _csv_rows(
        payload,
        label="rolling bucket CSV",
        required_columns={
            "cutoff_utc",
            "bucket",
            "metric_type",
            "hours",
            "baseline_mae_eur_mwh",
            "adjusted_mae_eur_mwh",
            "mae_improvement_eur_mwh",
            "baseline_rmse_eur_mwh",
            "adjusted_rmse_eur_mwh",
            "rmse_improvement_eur_mwh",
            "adjusted_error_p95_eur_mwh",
        },
    )
    if not isinstance(eligible_cutoffs, set) or not eligible_cutoffs:
        raise ValueError("eligible rolling cutoffs are invalid")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        composite = (row["cutoff_utc"], row["bucket"], row["metric_type"])
        if composite in seen:
            raise ValueError("rolling bucket CSV contains a duplicate fold/bucket row")
        seen.add(composite)
        groups.setdefault((row["bucket"], row["metric_type"]), []).append(row)
    if not groups:
        raise ValueError("rolling bucket CSV must contain at least one bucket")

    result: dict[str, dict[str, object]] = {}
    for (bucket, metric_type), group in sorted(groups.items()):
        if {row["cutoff_utc"] for row in group} != eligible_cutoffs:
            raise ValueError("rolling bucket coverage does not match eligible fold cutoffs")

        def values(column: str) -> list[float]:
            return [_csv_float(row[column], label=f"rolling bucket {column}") for row in group]

        improvements = values("mae_improvement_eur_mwh")
        key = f"{metric_type}:{bucket}"
        result[key] = {
            "fold_count": len(group),
            "total_hours": sum(
                _csv_int(row["hours"], label="rolling bucket hours") for row in group
            ),
            "mean_baseline_mae_eur_mwh": _mean_values(values("baseline_mae_eur_mwh")),
            "mean_adjusted_mae_eur_mwh": _mean_values(values("adjusted_mae_eur_mwh")),
            "mean_mae_improvement_eur_mwh": _mean_values(improvements),
            "positive_mae_improvement_fold_count": sum(value > 0.0 for value in improvements),
            "mean_baseline_rmse_eur_mwh": _mean_values(values("baseline_rmse_eur_mwh")),
            "mean_adjusted_rmse_eur_mwh": _mean_values(values("adjusted_rmse_eur_mwh")),
            "mean_rmse_improvement_eur_mwh": _mean_values(values("rmse_improvement_eur_mwh")),
            "mean_adjusted_error_p95_eur_mwh": _mean_values(values("adjusted_error_p95_eur_mwh")),
        }
    return result


def _require_declared_aggregate_matches(
    *,
    declared: object,
    recomputed: Mapping[str, object],
    label: str,
) -> None:
    actual = _mapping(declared, label=label)
    if set(actual) != set(recomputed):
        raise ValueError(f"{label} keys do not match captured CSV recomputation")
    for key, expected in recomputed.items():
        observed = actual.get(key)
        if isinstance(expected, dict):
            _require_declared_aggregate_matches(
                declared=observed,
                recomputed=expected,
                label=f"{label} {key}",
            )
        elif isinstance(expected, int):
            if isinstance(observed, bool) or not isinstance(observed, int) or observed != expected:
                raise ValueError(f"{label} {key} does not match captured CSV recomputation")
        else:
            selected = _finite_number(observed, label=f"{label} {key}")
            if not math.isclose(selected, float(expected), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{label} {key} does not match captured CSV recomputation")


def _csv_rows(
    payload: bytes,
    *,
    label: str,
    required_columns: set[str],
) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames
    if fields is None or len(fields) != len(set(fields)) or not required_columns.issubset(fields):
        raise ValueError(f"{label} schema is invalid")
    rows: list[dict[str, str]] = []
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"{label} row shape is invalid")
        rows.append({str(key): str(value) for key, value in raw.items()})
    return rows


def _csv_float(value: str, *, label: str) -> float:
    try:
        selected = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _csv_int(value: str, *, label: str) -> int:
    selected = _csv_float(value, label=label)
    if not selected.is_integer() or selected < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(selected)


def _csv_bool(value: str, *, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{label} must be True or False")


def _utc_datetime(value: str, *, label: str) -> datetime:
    try:
        selected = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if selected.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    return selected.astimezone(UTC)


def _mean_values(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty metric set")
    return math.fsum(values) / len(values)


def _quality_gates(
    *,
    top: Mapping[str, object],
    locked: Mapping[str, object],
    audit: Mapping[str, object],
    spot: Mapping[str, object],
) -> dict[str, bool]:
    coverage = _mapping(locked.get("coverage"), label="coverage")
    provenance = _mapping(locked.get("spot_provenance"), label="spot provenance")
    criteria = _mapping(audit.get("criteria"), label="audit criteria")
    return {
        "attempt_and_capture_sealed": bool(top.get("attempt_seal_sha256"))
        and bool(top.get("capture_seal_sha256")),
        "coverage_complete": coverage.get("missing_holdout_hours") == 0
        and not coverage.get("blocking_checks"),
        "future_holdout_pass": top.get("holdout_pass") is True
        and locked.get("holdout_pass") is True
        and audit.get("holdout_pass") is True,
        "independent_audit_pass": audit.get("status") == "LOCKED_HOLDOUT_PASS",
        "no_ompex": all(top.get(key) is False for key in _ompex_keys()),
        "provider_raw_replay_bound": provenance.get("pass") is True,
        "rolling_origin_no_leak": _mapping(
            spot.get("strict_lab_checks"), label="strict lab checks"
        ).get("rolling_folds_no_temporal_leak")
        is True,
        "rolling_origin_unique_ordered_cutoffs": _mapping(
            spot.get("strict_lab_checks"), label="strict lab checks"
        ).get("rolling_folds_unique_ordered_cutoffs")
        is True,
        "rolling_origin_non_overlapping_evaluations": _mapping(
            spot.get("strict_lab_checks"), label="strict lab checks"
        ).get("rolling_folds_non_overlapping_evaluations")
        is True,
        "strict_lab_gate_pass": locked.get("strict_lab_gate_pass") is True
        and criteria.get("strict_lab_gate_pass") is True
        and spot.get("strict_lab_gate_pass") is True,
    }


def _bucket_metrics(value: object) -> dict[str, dict[str, object]]:
    raw = _mapping(value, label="rolling bucket metrics")
    result: dict[str, dict[str, object]] = {}
    for name in sorted(raw):
        bucket = _mapping(raw[name], label=f"rolling bucket {name}")
        baseline = _finite_number(
            bucket.get("mean_baseline_mae_eur_mwh"),
            label=f"{name} baseline MAE",
        )
        adjusted = _finite_number(
            bucket.get("mean_adjusted_mae_eur_mwh"),
            label=f"{name} adjusted MAE",
        )
        improvement = _finite_number(
            bucket.get("mean_mae_improvement_eur_mwh"),
            label=f"{name} MAE improvement",
        )
        result[name] = {
            "fold_count": int(_finite_number(bucket.get("fold_count"), label=f"{name} folds")),
            "positive_improvement_fold_count": int(
                _finite_number(
                    bucket.get("positive_mae_improvement_fold_count"),
                    label=f"{name} positive folds",
                )
            ),
            "baseline_mae_eur_mwh": baseline,
            "adjusted_mae_eur_mwh": adjusted,
            "mae_improvement_eur_mwh": improvement,
            "relative_mae_improvement_percent": _percentage(improvement, baseline),
        }
    return result


def _metric_mapping(value: object, *, label: str) -> dict[str, float]:
    raw = _mapping(value, label=label)
    return {
        key: _finite_number(raw.get(key), label=f"{label} {key}")
        for key in (
            "baseline_residual_mae_eur_mwh",
            "adjusted_residual_mae_eur_mwh",
            "residual_mae_improvement_eur_mwh",
        )
    }


def _read_bound_json(
    document: Mapping[str, object],
    *,
    path_key: str,
    sha_key: str,
    run_root: Path,
    label: str,
    bound: dict[str, dict[str, object]],
) -> dict[str, object]:
    payload = _read_bound_bytes(
        document,
        path_key=path_key,
        sha_key=sha_key,
        allowed_root=run_root,
        label=label,
        bound=bound,
        max_bytes=_MAX_JSON_BYTES,
    )
    return _strict_mapping(payload, label=label)


def _read_bound_bytes(
    document: Mapping[str, object],
    *,
    path_key: str,
    sha_key: str,
    allowed_root: Path,
    label: str,
    bound: dict[str, dict[str, object]],
    max_bytes: int = _MAX_BOUND_ARTIFACT_BYTES,
) -> bytes:
    path = Path(str(document.get(path_key, "")))
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    selected = assert_absolute_path_has_no_links(path)
    if not _is_relative_to(selected, allowed_root):
        raise ValueError(f"{label} escapes its admitted root")
    payload = _read_stable(selected, max_bytes=max_bytes)
    expected = _sha256_value(document.get(sha_key), label=f"{label} SHA-256")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    bound[label] = {
        "relative_path": selected.relative_to(allowed_root).as_posix(),
        "sha256": actual,
        "size_bytes": len(payload),
    }
    return payload


def _read_repo_bound_bytes(
    document: Mapping[str, object],
    *,
    path_key: str,
    sha_key: str,
    repo_root: Path,
    label: str,
    bound: dict[str, dict[str, object]],
) -> bytes:
    raw = Path(str(document.get(path_key, "")))
    selected = raw if raw.is_absolute() else repo_root / raw
    selected = assert_absolute_path_has_no_links(selected)
    if not _is_relative_to(selected, repo_root):
        raise ValueError(f"{label} escapes the repository root")
    payload = _read_stable(selected, max_bytes=_MAX_BOUND_ARTIFACT_BYTES)
    expected = _sha256_value(document.get(sha_key), label=f"{label} SHA-256")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    bound[label] = {
        "relative_path": selected.relative_to(repo_root).as_posix(),
        "sha256": actual,
        "size_bytes": len(payload),
    }
    return payload


def _admit_output(path: Path, *, run_root: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("local quality output path must be absolute")
    parent = assert_absolute_path_has_no_links(path.parent)
    if not parent.is_dir():
        raise ValueError("local quality output parent must already exist")
    selected = parent / path.name
    if _is_relative_to(selected, run_root) or _is_relative_to(run_root, selected):
        raise ValueError("local quality output must be outside locked T057 evidence")
    return selected


def _read_stable(path: Path, *, max_bytes: int) -> bytes:
    payload = read_stable_single_link_file(
        path,
        label=f"local quality source {path.name}",
        max_bytes=max_bytes,
    )
    if not payload:
        raise ValueError(f"local quality source size is invalid: {path.name}")
    return payload


def _require_positive_uplift(
    baseline: float,
    adjusted: float,
    improvement: float,
    *,
    label: str,
) -> None:
    if not math.isclose(baseline - adjusted, improvement, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} improvement is arithmetically inconsistent")
    if improvement <= 0.0 or adjusted >= baseline:
        raise ValueError(f"{label} improvement must be strictly positive")


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_mapping,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    return _mapping(value, label=label)


def _unique_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _sha256_value(value: object, *, label: str) -> str:
    selected = str(value)
    if len(selected) != _SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _percentage(numerator: float, denominator: float) -> str:
    if denominator == 0.0:
        raise ValueError("local quality percentage denominator cannot be zero")
    try:
        value = Decimal(str(numerator)) * Decimal("100") / Decimal(str(denominator))
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise ValueError("local quality percentage is invalid") from exc
    return format(value.quantize(Decimal("0.000001")), "f")


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _render_markdown(report: Mapping[str, object]) -> str:
    coverage = _mapping(report["coverage"], label="report coverage")
    holdout = _mapping(report["holdout"], label="report holdout")
    rolling = _mapping(report["rolling_origin"], label="report rolling origin")
    buckets = _mapping(report["bucket_metrics"], label="report bucket metrics")
    gates = _mapping(report["gates"], label="report gates")
    assessment = _mapping(report["assessment"], label="report assessment")
    lines = [
        "# Qualité locale PFC suisse - T057",
        "",
        f"Statut : **{report['status']}**",
        "",
        "Ce rapport est diagnostique, non promotionnel et lié par hash aux preuves T057.",
        "",
        "## Résultat principal",
        "",
        "| Mesure | Baseline | Adjusted | Amélioration | Relative |",
        "|---|---:|---:|---:|---:|",
        (
            "| Holdout futur MAE | "
            f"{holdout['baseline_mae_eur_mwh']:.3f} | "
            f"{holdout['adjusted_mae_eur_mwh']:.3f} | "
            f"{holdout['mae_improvement_eur_mwh']:.3f} | "
            f"{holdout['relative_mae_improvement_percent']}% |"
        ),
        (
            "| Rolling-origin profil MAE | "
            f"{rolling['baseline_profile_mae_eur_mwh']:.3f} | "
            f"{rolling['adjusted_profile_mae_eur_mwh']:.3f} | "
            f"{rolling['mae_improvement_eur_mwh']:.3f} | "
            f"{rolling['relative_mae_improvement_percent']}% |"
        ),
        "",
        f"Holdout : {coverage['holdout_hours']} heures, "
        f"{coverage['missing_holdout_hours']} manquante(s).",
        "",
        f"Rolling-origin : {rolling['positive_improvement_fold_count']}/"
        f"{rolling['fold_count']} folds améliorent la MAE.",
        "",
        "## Segments",
        "",
        "| Segment | Baseline MAE | Adjusted MAE | Amélioration | Relative | Folds + |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(buckets):
        bucket = _mapping(buckets[name], label=f"report bucket {name}")
        lines.append(
            f"| {name} | {bucket['baseline_mae_eur_mwh']:.3f} | "
            f"{bucket['adjusted_mae_eur_mwh']:.3f} | "
            f"{bucket['mae_improvement_eur_mwh']:.3f} | "
            f"{bucket['relative_mae_improvement_percent']}% | "
            f"{bucket['positive_improvement_fold_count']}/{bucket['fold_count']} |"
        )
    lines.extend(["", "## Gates", ""])
    for name in sorted(gates):
        lines.append(f"- {'PASS' if gates[name] else 'FAIL'} - `{name}`")
    lines.extend(
        [
            "",
            "## Lecture honnête",
            "",
            f"- Force : {assessment['main_strength']}.",
            f"- Limite : {assessment['main_limit']}.",
            "- La matérialité économique n'est pas établie par le seuil T057.",
            "- Production : **NON AUTORISÉE**.",
            "",
        ]
    )
    return "\n".join(lines)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ompex_keys() -> tuple[str, str, str]:
    return ("ompex_used_in_backtest", "ompex_used_in_model", "ompex_used_in_selection")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--expected-run-summary-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_local_quality_report(
        run_summary=args.run_summary,
        expected_run_summary_sha256=args.expected_run_summary_sha256,
        repo_root=args.repo_root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
