"""Evidence-bound quality status for the current non-production CH LT candidate."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation.dependence_power_supersession import (
    REGISTRY_RELATIVE_PATH as DEPENDENCE_REGISTRY_RELATIVE_PATH,
)
from pfc_shaping.validation.dependence_power_supersession import (
    REPLACEMENT_RELATIVE_PATH,
    REPLACEMENT_SHA256,
    verify_dependence_power_selection,
)
from scripts.verify_t057_evidence_supersession import verify_supersession_registry

SCHEMA_VERSION = "ch_lt_local_candidate_quality.v2"
STATUS = "LOCAL_STRUCTURAL_DIAGNOSTIC_ONLY_NO_GO"
CANONICAL_COMPLETION_RELATIVE_PATH = (
    "output/phase14/local_pfc_prospective_intraday_20260724_run15/completion_manifest.json"
)
CANONICAL_COMPLETION_SHA256 = (
    "7ff70c65f49d53f0f1ceba29dacf71826df45547bf3c334fd327b3d03573540d"
)
ROLLING_SUMMARY_RELATIVE_PATH = (
    "output/phase14/intraday_shape_rolling_origin_20260724_v3/summary.json"
)
ROLLING_SUMMARY_SHA256 = "523169da91a3ec3ab46f341608274a449f035488bf671d2af82481112a3d8f09"
ROLLING_FOLDS_RELATIVE_PATH = (
    "output/phase14/intraday_shape_rolling_origin_20260724_v3/fold_metrics.csv"
)
ROLLING_FOLDS_SHA256 = "910dd2dbe0a2c87f345f05431be14769d45e4113e1f54301176ba25fc7ecfe45"
DEPENDENCE_REGISTRY_SHA256 = (
    "9fd9cf706c768716a42967962527a49a9f92c7a16d1e60de0b3d910565312b72"
)
T057_REGISTRY_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/T057-EVIDENCE-SUPERSESSION-REGISTRY.json"
)

_MAX_BOUND_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_OUTPUT_BYTES = 128 * 1024 * 1024
_EXPECTED_SCENARIOS = ("slow", "central", "fast")
_EXPECTED_COLUMNS = (
    "price_shape",
    "B",
    "f_S",
    "f_W",
    "f_H",
    "f_Q",
    "f_WV",
    "delta_wv",
    "f_bridge",
    "profile_type",
    "confidence",
    "calibrated",
    "p10",
    "p90",
)


class ChLtLocalCandidateQualityError(ValueError):
    """Raised when current local candidate evidence is incomplete or inconsistent."""


def build_local_candidate_quality_report(
    *,
    repo_root: str | Path,
    completion_manifest: str | Path,
    expected_completion_sha256: str,
    output_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, object]:
    """Verify current evidence and emit an explicitly non-promotional status."""

    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    manifest_path = assert_absolute_path_has_no_links(_absolute(completion_manifest))
    canonical_manifest = _repo_path(root, CANONICAL_COMPLETION_RELATIVE_PATH)
    if _path_key(manifest_path) != _path_key(canonical_manifest):
        raise ChLtLocalCandidateQualityError("completion manifest path is not canonical run15")
    manifest_payload = _stable(manifest_path, label="run15 completion manifest")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if manifest_sha256 != expected_completion_sha256:
        raise ChLtLocalCandidateQualityError("completion manifest caller-held SHA-256 mismatch")
    if expected_completion_sha256 != CANONICAL_COMPLETION_SHA256:
        raise ChLtLocalCandidateQualityError("completion manifest is not canonical run15 bytes")
    manifest = _strict_mapping(manifest_payload, label="completion manifest")
    _verify_manifest_authority(manifest)

    run_root = manifest_path.parent
    output_payloads = _verify_output_inventory(manifest, root=root, run_root=run_root)
    input_state = _verify_input_inventory(manifest, root=root)
    monthly_manifest = _strict_mapping(
        output_payloads["monthly_curve_manifest"], label="monthly curve manifest"
    )
    panel_manifest = _read_bound_json(
        root,
        "output/phase14/prospective_intraday_panel_20260724/panel-manifest.json",
        "af8232e7045eb9aa685d946770a56a108e438a6a71e9cbdab47614abcc5a9e4f",
        label="intraday panel manifest",
    )
    _verify_monthly_and_panel_authority(monthly_manifest, panel_manifest)

    scenario_frames = {
        scenario: _read_candidate_frame(output_payloads[f"pfc_{scenario}"], scenario=scenario)
        for scenario in _EXPECTED_SCENARIOS
    }
    scenario_quality = _analyze_scenarios(
        scenario_frames,
        monthly_solution=_mapping(
            monthly_manifest.get("monthly_solution"), label="monthly solution"
        ),
    )
    rolling_quality = _rolling_origin_quality(root)
    t057 = verify_supersession_registry(
        registry=_repo_path(root, T057_REGISTRY_RELATIVE_PATH),
        repo_root=root,
    )
    t057_claims = _mapping(t057.get("effective_claims"), label="T057 effective claims")

    solver_kkt = _mapping(monthly_manifest.get("solver_kkt"), label="solver KKT")
    gates = {
        "completion_and_output_bytes_verified": True,
        "monthly_solver_is_level_authority": True,
        "monthly_level_preserved_within_1e_9_eur_mwh": bool(
            scenario_quality["maximum_monthly_mean_residual_eur_mwh"] <= 1e-9
        ),
        "monthly_solver_constraint_residual_within_1e_9_eur_mwh": bool(
            float(solver_kkt["max_abs_constraint_residual"]) <= 1e-9
        ),
        "governed_fresh_hard_ch_quotes": False,
        "exact_production_eex_repricing_proven": False,
        "direct_governed_ch_15min_truth": False,
        "de_lu_to_ch_transfer_validated": False,
        "run_code_reproducible_from_bound_bytes": False,
        "rolling_origin_ch_quality_established": False,
        "t057_historical_robustness_established": False,
        "future_holdout_sufficiency_established": False,
        "probabilistic_quantiles_calibrated": False,
        "economic_materiality_established": False,
        "promotion_eligible": False,
    }
    if not all(
        gates[name]
        for name in (
            "completion_and_output_bytes_verified",
            "monthly_solver_is_level_authority",
            "monthly_level_preserved_within_1e_9_eur_mwh",
            "monthly_solver_constraint_residual_within_1e_9_eur_mwh",
        )
    ):
        raise ChLtLocalCandidateQualityError("local structural invariants did not pass")

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "as_of_date": "2026-07-27",
        "market": "CH",
        "production_authorization": False,
        "promotion_gate": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "t057_supersession_metadata_read": True,
        "t057_reused_for_selection": False,
        "source": {
            "completion_manifest_path": CANONICAL_COMPLETION_RELATIVE_PATH,
            "completion_manifest_sha256": manifest_sha256,
            "completion_manifest_last_written": True,
            "code_execution_attestation": manifest.get("code_execution_attestation"),
            "output_role_count": len(output_payloads),
            "all_output_bytes_verified": True,
            "input_state": input_state,
        },
        "curve": scenario_quality,
        "monthly_solver": {
            "level_authority": "solver",
            "source_kind": "TEST_FIXTURE",
            "hard_quote_eligible": False,
            "promotion_eligible": False,
            "constraint_count": len(monthly_manifest.get("constraint_provenance_rows", [])),
            "max_abs_constraint_residual": float(
                solver_kkt["max_abs_constraint_residual"]
            ),
            "stationarity_residual": float(solver_kkt["stationarity_residual"]),
            "condition_number": float(solver_kkt["condition_number"]),
            "structural_prior_status": monthly_manifest.get("structural_status"),
        },
        "intraday": {
            "source_market": "DE-LU_PROXY_FOR_CH_LOCAL_DIAGNOSTIC",
            "source_authority": panel_manifest.get("status"),
            "panel_rows": panel_manifest.get("rows"),
            "fallback_cell_share_percent": 61.46,
            "directly_learned_cells": 185,
            "fallback_cells": 295,
            "cell_diagnostics_source": (
                "BOUND_RUN15_QUALITY_SUMMARY_COPIED_NOT_INDEPENDENTLY_RECOMPUTED"
            ),
            "transfer_to_ch_validated": False,
            "rolling_origin": rolling_quality,
        },
        "t057_corrected_evidence": {
            "supersession_registry_sha256": t057.get("registry_sha256"),
            "correction_sha256": t057.get("authoritative_correction_sha256"),
            "historical_distinct_fold_count": t057_claims.get(
                "historical_distinct_fold_count"
            ),
            "historical_robustness_established": False,
            "future_holdout_episode_count": t057_claims.get("future_holdout_episode_count"),
            "future_holdout_hours": t057_claims.get("future_holdout_hours"),
            "materiality_established": False,
        },
        "gates": gates,
        "assessment": {
            "usable_now_for": [
                "LOCAL_VISUAL_INSPECTION",
                "STRUCTURAL_MONTHLY_LEVEL_DIAGNOSTICS",
                "PIPELINE_SMOKE_TESTS",
            ],
            "not_usable_for": [
                "PRODUCTION_FMV",
                "MODEL_SELECTION",
                "SCIENTIFIC_CH_15MIN_CLAIMS",
                "PROBABILISTIC_RISK_VALUATION",
                "PROMOTION",
            ],
            "required_next_evidence": [
                "FRESH_GOVERNED_CH_EEX_POINT_IN_TIME_SNAPSHOT",
                "DIRECT_GOVERNED_CH_TRUTH_OR_EXPLICIT_UNSUPPORTED_15MIN",
                "NEW_HASH_BOUND_REPRODUCIBLE_CANDIDATE_BUILD",
                "PREREGISTERED_UNIQUE_ROLLING_ORIGINS_EXCLUDING_PILOT_REUSE",
                "MULTIPLE_UNTOUCHED_FUTURE_HOLDOUT_EPISODES",
                "CALIBRATED_PROBABILISTIC_SCENARIOS_AND_FMV_MATERIALITY",
            ],
        },
    }
    json_bytes = _canonical_json(report)
    markdown_bytes = _render_markdown(report).encode("utf-8")
    json_path = _output_path(output_json, run_root=run_root)
    markdown_path = _output_path(output_markdown, run_root=run_root)
    if _path_key(json_path) == _path_key(markdown_path):
        raise ChLtLocalCandidateQualityError("quality outputs must be distinct")
    write_durable_exact_bytes(json_path, json_bytes, label="CH LT local quality JSON")
    write_durable_exact_bytes(markdown_path, markdown_bytes, label="CH LT local quality Markdown")
    return report


def _verify_manifest_authority(manifest: Mapping[str, object]) -> None:
    expected = {
        "schema_version": "local_test_ch_pfc_completion.v1",
        "status": "COMPLETE_LOCAL_TEST_NO_GO",
        "completion_manifest_last_written": True,
        "forward_source_kind": "TEST_FIXTURE",
        "hard_quote_eligible": False,
        "monthly_level_authority": "solver",
        "monthly_manifest_promotion_eligible": False,
        "production_authorization": False,
        "promotion_gate": False,
        "code_execution_attestation": (
            "NOT_PROVIDED_SOURCE_HASHES_ARE_OBSERVED_AFTER_IMPORT_ONLY"
        ),
    }
    for field, value in expected.items():
        _equal(manifest.get(field), value, f"completion {field}")


def _verify_output_inventory(
    manifest: Mapping[str, object], *, root: Path, run_root: Path
) -> dict[str, bytes]:
    outputs = _mapping(manifest.get("outputs"), label="completion outputs")
    expected_roles = {
        "expanded_scenarios",
        "governance_report",
        "monthly_curve_manifest",
        "pfc_central",
        "pfc_fast",
        "pfc_slow",
        "quality_summary",
        "scenario_features",
        "structural_fan_chart",
    }
    if set(outputs) != expected_roles:
        raise ChLtLocalCandidateQualityError("completion output roles are not exact")
    captured: dict[str, bytes] = {}
    captured_bytes = 0
    for role in sorted(outputs):
        record = _mapping(outputs[role], label=f"output {role}")
        path = _manifest_path(root, record.get("path"), label=f"output {role}")
        if not _is_within(path, run_root):
            raise ChLtLocalCandidateQualityError(f"output {role} escapes the run root")
        payload = _stable(path, label=f"output {role}")
        captured_bytes += len(payload)
        if captured_bytes > _MAX_TOTAL_OUTPUT_BYTES:
            raise ChLtLocalCandidateQualityError(
                "completion output inventory exceeds cumulative byte budget"
            )
        _verify_record(payload, record, label=f"output {role}")
        captured[role] = payload
    return captured


def _verify_input_inventory(
    manifest: Mapping[str, object], *, root: Path
) -> dict[str, object]:
    inputs = _mapping(manifest.get("inputs"), label="completion inputs")
    code_drift: list[dict[str, object]] = []
    verified_data_roles: list[str] = []
    for role in sorted(inputs):
        record = _mapping(inputs[role], label=f"input {role}")
        path = _manifest_path(root, record.get("path"), label=f"input {role}")
        payload = _stable(path, label=f"input {role}")
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        expected_sha256 = str(record.get("sha256", ""))
        actual_size = len(payload)
        expected_size = record.get("size_bytes")
        if role.endswith("_source_observed_after_import"):
            if actual_sha256 != expected_sha256 or actual_size != expected_size:
                code_drift.append(
                    {
                        "role": role,
                        "path": str(record.get("path")),
                        "run_observed_sha256": expected_sha256,
                        "current_sha256": actual_sha256,
                        "run_observed_size_bytes": expected_size,
                        "current_size_bytes": actual_size,
                    }
                )
            continue
        _verify_record(payload, record, label=f"input {role}")
        verified_data_roles.append(role)
    return {
        "verified_non_code_input_roles": verified_data_roles,
        "source_drift": code_drift,
        "source_drift_count": len(code_drift),
        "exact_run_code_bytes_captured": False,
        "reproducible_execution_proven": False,
    }


def _verify_monthly_and_panel_authority(
    monthly: Mapping[str, object], panel: Mapping[str, object]
) -> None:
    expected_monthly = {
        "market": "CH",
        "monthly_level_authority": "solver",
        "forward_source_kind": "TEST_FIXTURE",
        "hard_quote_eligible": False,
        "promotion_eligible": False,
        "structural_status": "STRUCTURAL_TEMPLATE",
    }
    for field, value in expected_monthly.items():
        _equal(monthly.get(field), value, f"monthly {field}")
    _equal(panel.get("status"), "COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO", "panel status")
    _equal(panel.get("market"), "DE-LU_PROXY_FOR_CH_LOCAL_DIAGNOSTIC", "panel market")
    _equal(panel.get("production_authorization"), False, "panel production")
    _equal(panel.get("promotion_gate"), False, "panel promotion")


def _read_candidate_frame(payload: bytes, *, scenario: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise ChLtLocalCandidateQualityError(f"{scenario} PFC parquet is invalid") from exc
    if tuple(frame.columns) != _EXPECTED_COLUMNS:
        raise ChLtLocalCandidateQualityError(f"{scenario} PFC columns are not exact")
    index = pd.DatetimeIndex(frame.index)
    if (
        index.tz is None
        or not index.is_monotonic_increasing
        or not index.is_unique
        or len(index) != 155_616
    ):
        raise ChLtLocalCandidateQualityError(f"{scenario} PFC timestamp axis is invalid")
    if len(index) > 1 and not np.all(np.diff(index.asi8) == 900 * 1_000_000_000):
        raise ChLtLocalCandidateQualityError(f"{scenario} PFC cadence is not 15 minutes")
    for column in ("price_shape", "B"):
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ChLtLocalCandidateQualityError(f"{scenario} PFC {column} is non-finite")
    if not bool(frame["calibrated"].all()):
        raise ChLtLocalCandidateQualityError(f"{scenario} PFC has uncalibrated rows")
    return frame


def _analyze_scenarios(
    frames: Mapping[str, pd.DataFrame], *, monthly_solution: Mapping[str, object]
) -> dict[str, object]:
    first_index = pd.DatetimeIndex(frames["slow"].index)
    if any(not first_index.equals(pd.DatetimeIndex(frames[name].index)) for name in frames):
        raise ChLtLocalCandidateQualityError("scenario timestamp axes differ")
    solution = pd.Series({key: float(value) for key, value in monthly_solution.items()})
    summaries: dict[str, object] = {}
    maximum_residual = 0.0
    all_quantiles_null = True
    for scenario in _EXPECTED_SCENARIOS:
        frame = frames[scenario]
        local = pd.DatetimeIndex(frame.index).tz_convert("Europe/Zurich")
        months = pd.Index(local.strftime("%Y-%m"))
        monthly_prices = frame["price_shape"].groupby(months).mean()
        monthly_b = frame["B"].groupby(months).mean()
        if not monthly_prices.index.equals(solution.index):
            raise ChLtLocalCandidateQualityError(
                f"{scenario} PFC monthly inventory differs from solver"
            )
        price_residual = float((monthly_prices - solution).abs().max())
        b_residual = float((monthly_b - solution).abs().max())
        maximum_residual = max(maximum_residual, price_residual, b_residual)
        p10_count = int(frame["p10"].notna().sum())
        p90_count = int(frame["p90"].notna().sum())
        all_quantiles_null = all_quantiles_null and p10_count == 0 and p90_count == 0
        price = frame["price_shape"].to_numpy(dtype=float)
        summaries[scenario] = {
            "rows": len(frame),
            "start_utc": first_index.min().isoformat(),
            "end_utc": first_index.max().isoformat(),
            "mean_eur_mwh": float(np.mean(price)),
            "min_eur_mwh": float(np.min(price)),
            "p05_eur_mwh": float(np.quantile(price, 0.05)),
            "p95_eur_mwh": float(np.quantile(price, 0.95)),
            "max_eur_mwh": float(np.max(price)),
            "monthly_price_residual_eur_mwh": price_residual,
            "monthly_base_residual_eur_mwh": b_residual,
            "p10_non_null_rows": p10_count,
            "p90_non_null_rows": p90_count,
        }
    return {
        "resolution_minutes": 15,
        "structural_display_weights_not_probabilities": {
            "slow": 0.25,
            "central": 0.5,
            "fast": 0.25,
        },
        "scenario_summaries": summaries,
        "maximum_monthly_mean_residual_eur_mwh": maximum_residual,
        "monthly_solver_level_preserved": maximum_residual <= 1e-9,
        "calibrated_probabilistic_quantiles_present": not all_quantiles_null,
        "scenario_interpretation": "STRUCTURAL_PATHS_NOT_CALIBRATED_PROBABILITIES",
    }


def _rolling_origin_quality(root: Path) -> dict[str, object]:
    selection = verify_dependence_power_selection(
        repo_root=root,
        registry_path=_repo_path(root, DEPENDENCE_REGISTRY_RELATIVE_PATH),
        expected_registry_sha256=DEPENDENCE_REGISTRY_SHA256,
        selected_artifact_path=_repo_path(root, REPLACEMENT_RELATIVE_PATH),
        expected_selected_artifact_sha256=REPLACEMENT_SHA256,
    )
    if selection.get("local_diagnostic_selection_current") is not True:
        raise ChLtLocalCandidateQualityError("dependence/power v2 selection is not current")
    summary = _read_bound_json(
        root,
        ROLLING_SUMMARY_RELATIVE_PATH,
        ROLLING_SUMMARY_SHA256,
        label="rolling-origin summary",
    )
    fold_payload = _read_bound_bytes(
        root,
        ROLLING_FOLDS_RELATIVE_PATH,
        ROLLING_FOLDS_SHA256,
        label="rolling-origin folds",
    )
    rows = list(csv.DictReader(io.StringIO(fold_payload.decode("utf-8"))))
    by_origin: dict[str, dict[str, float]] = {}
    by_model: dict[str, list[tuple[int, float, float]]] = {}
    for row in rows:
        origin = row["origin_utc"]
        model = row["model"]
        by_origin.setdefault(origin, {})[model] = float(row["mae_eur_mwh"])
        by_model.setdefault(model, []).append(
            (
                int(row["n_quarter_hours"]),
                float(row["mae_eur_mwh"]),
                float(row["rmse_eur_mwh"]),
            )
        )
    if len(by_origin) != 16 or any(len(models) != 2 for models in by_origin.values()):
        raise ChLtLocalCandidateQualityError("rolling-origin pair inventory is invalid")
    candidate = "candidate_sparse_price_space_24_dense_envelope"
    benchmark = "incumbent_ratio"
    differences = [models[benchmark] - models[candidate] for models in by_origin.values()]
    tolerance = 1e-12
    wins = sum(value > tolerance for value in differences)
    losses = sum(value < -tolerance for value in differences)
    ties = len(differences) - wins - losses
    metrics = _mapping(summary.get("metrics"), label="rolling metrics")
    candidate_metrics = _mapping(metrics.get(candidate), label="candidate metrics")
    benchmark_metrics = _mapping(metrics.get(benchmark), label="benchmark metrics")
    def _recompute(model: str) -> tuple[float, float]:
        model_rows = by_model.get(model, [])
        total = sum(count for count, _, _ in model_rows)
        if len(model_rows) != 16 or total <= 0:
            raise ChLtLocalCandidateQualityError(
                f"rolling-origin {model} fold inventory is invalid"
            )
        mae = sum(count * value for count, value, _ in model_rows) / total
        rmse = math.sqrt(
            sum(count * value**2 for count, _, value in model_rows) / total
        )
        return mae, rmse

    candidate_mae, candidate_rmse = _recompute(candidate)
    benchmark_mae, benchmark_rmse = _recompute(benchmark)
    for actual, reported, label in (
        (candidate_mae, float(candidate_metrics["mae_eur_mwh"]), "candidate MAE"),
        (candidate_rmse, float(candidate_metrics["rmse_eur_mwh"]), "candidate RMSE"),
        (benchmark_mae, float(benchmark_metrics["mae_eur_mwh"]), "benchmark MAE"),
        (benchmark_rmse, float(benchmark_metrics["rmse_eur_mwh"]), "benchmark RMSE"),
    ):
        if not np.isclose(actual, reported, rtol=0.0, atol=1e-12):
            raise ChLtLocalCandidateQualityError(
                f"rolling-origin {label} replay differs from bound summary"
            )
    if (wins, losses, ties) != (6, 6, 4):
        raise ChLtLocalCandidateQualityError("rolling-origin win/loss/tie replay differs")
    if str(summary.get("status")) != "LOCAL_DIAGNOSTIC_CANDIDATE_REJECTED_NOT_PRODUCTION":
        raise ChLtLocalCandidateQualityError("rolling-origin candidate status is not rejected")
    latest = by_origin[sorted(by_origin)[-1]]
    return {
        "target": "DE_LU_QUARTER_HOUR_PRICE_GIVEN_REALIZED_PARENT_HOUR_MEAN",
        "target_is_full_ch_lt": False,
        "confirmatory_reuse_forbidden": True,
        "aggregate_metrics_source": "RECOMPUTED_FROM_HASH_BOUND_FOLD_ROWS",
        "origin_count": 16,
        "candidate_mae_eur_mwh": candidate_mae,
        "incumbent_mae_eur_mwh": benchmark_mae,
        "candidate_rmse_eur_mwh": candidate_rmse,
        "incumbent_rmse_eur_mwh": benchmark_rmse,
        "relative_mae_improvement": (benchmark_mae - candidate_mae) / benchmark_mae,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "latest_candidate_mae_eur_mwh": latest[candidate],
        "latest_incumbent_mae_eur_mwh": latest[benchmark],
        "latest_fold_noninferior": latest[candidate] <= latest[benchmark] + tolerance,
        "candidate_accepted": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def _read_bound_json(
    root: Path, relative_path: str, sha256: str, *, label: str
) -> Mapping[str, object]:
    return _strict_mapping(
        _read_bound_bytes(root, relative_path, sha256, label=label), label=label
    )


def _read_bound_bytes(root: Path, relative_path: str, sha256: str, *, label: str) -> bytes:
    payload = _stable(_repo_path(root, relative_path), label=label)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ChLtLocalCandidateQualityError(f"{label} SHA-256 mismatch")
    return payload


def _verify_record(payload: bytes, record: Mapping[str, object], *, label: str) -> None:
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise ChLtLocalCandidateQualityError(f"{label} SHA-256 mismatch")
    if len(payload) != record.get("size_bytes"):
        raise ChLtLocalCandidateQualityError(f"{label} size mismatch")


def _stable(path: Path, *, label: str) -> bytes:
    return read_stable_single_link_file(path, label=label, max_bytes=_MAX_BOUND_BYTES)


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtLocalCandidateQualityError(f"{label} is invalid strict JSON") from exc
    return _mapping(value, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtLocalCandidateQualityError(f"{label} must be an object")
    return value


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtLocalCandidateQualityError(f"{label} is invalid")


def _manifest_path(root: Path, raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ChLtLocalCandidateQualityError(f"{label} path is invalid")
    path = Path(raw)
    selected = path if path.is_absolute() else root / path
    selected = assert_absolute_path_has_no_links(Path(os.path.abspath(selected)))
    if not _is_within(selected, root):
        raise ChLtLocalCandidateQualityError(f"{label} path escapes the repository")
    return selected


def _repo_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ChLtLocalCandidateQualityError("repository relative path is unsafe")
    return assert_absolute_path_has_no_links(Path(os.path.abspath(root / path)))


def _output_path(value: str | Path, *, run_root: Path) -> Path:
    path = assert_absolute_path_has_no_links(_absolute(value))
    if _is_within(path, run_root):
        raise ChLtLocalCandidateQualityError("quality output must be outside immutable run15")
    if not path.parent.is_dir():
        raise ChLtLocalCandidateQualityError("quality output directory is unavailable")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _absolute(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    return selected if selected.is_absolute() else Path(os.path.abspath(selected))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _render_markdown(report: Mapping[str, object]) -> str:
    curve = _mapping(report["curve"], label="curve report")
    summaries = _mapping(curve["scenario_summaries"], label="scenario summaries")
    rolling = _mapping(
        _mapping(report["intraday"], label="intraday report")["rolling_origin"],
        label="rolling report",
    )
    t057 = _mapping(report["t057_corrected_evidence"], label="T057 report")
    lines = [
        "# CH LT local candidate quality — 2026-07-27",
        "",
        f"Status: `{report['status']}`  ",
        "Production: `NO_GO`  ",
        "Promotion: `false`",
        "",
        "## What is proven",
        "",
        "- All run15 output bytes are hash-verified.",
        f"- Monthly solver preservation residual: `{curve['maximum_monthly_mean_residual_eur_mwh']:.3e} EUR/MWh`.",
        "- The monthly solver remains the sole level authority.",
        "",
        "## Curve statistics",
        "",
        "| scenario | mean | min | p05 | p95 | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in _EXPECTED_SCENARIOS:
        item = _mapping(summaries[scenario], label=f"{scenario} summary")
        lines.append(
            f"| {scenario} | {item['mean_eur_mwh']:.4f} | {item['min_eur_mwh']:.4f} | "
            f"{item['p05_eur_mwh']:.4f} | {item['p95_eur_mwh']:.4f} | "
            f"{item['max_eur_mwh']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Quality verdict",
            "",
            f"- DE-LU pilot MAE: candidate `{rolling['candidate_mae_eur_mwh']:.6f}` vs incumbent `{rolling['incumbent_mae_eur_mwh']:.6f}` EUR/MWh.",
            f"- Paired origins: `{rolling['wins']}` wins, `{rolling['losses']}` losses, `{rolling['ties']}` ties; latest fold noninferior: `{str(rolling['latest_fold_noninferior']).lower()}`.",
            "- The challenger remains rejected and the DE-LU pilot cannot be reused as CH confirmatory evidence.",
            f"- Corrected T057: `{t057['historical_distinct_fold_count']}` distinct historical fold and `{t057['future_holdout_episode_count']}` future episode; robustness and materiality are not established.",
            "- P10/P90 are absent; the three paths are structural scenarios, not calibrated probabilities.",
            "- The build used fixture forwards and a mixed-authority DE-LU proxy; exact production EEX repricing and direct CH 15-minute quality are unproven.",
            "- Exact run code bytes were not captured, and one observed source has since drifted; reproducible execution is not proven.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CANONICAL_COMPLETION_RELATIVE_PATH",
    "CANONICAL_COMPLETION_SHA256",
    "ChLtLocalCandidateQualityError",
    "SCHEMA_VERSION",
    "STATUS",
    "build_local_candidate_quality_report",
]
