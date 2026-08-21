"""Build a packaged truth-blind, non-countable CH LT prediction commitment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json
from pfc_shaping.validation import ch_lt_origin_target_mask_inventory as origin_inventory

# Capture the governed invocation root once without embedding a workstation path.
ROOT = Path.cwd()
SELECTED_INVENTORY_RELATIVE = Path(
    "build/ch-lt-origin-target-mask-inventories/"
    "structural-dry-run-20260731T072027Z-schema-v2.json"
)
SELECTED_INVENTORY_SHA256 = (
    "a4d5722841df549ba24ca2dd190fa0f5b1d79738fc63680acf93034e148419b7"
)
CANDIDATE_RELATIVE = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-SUCCESSOR-CANDIDATE-CORE-V3-20260730.json"
)
CANDIDATE_SHA256 = "e2c2a6f9d3ca677991f3f76f03dec1328492e2f60a4015b7796a2ff6aa6f905a"
SELECTION_RELATIVE = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-LOCAL-QUALITY-CURRENT-SELECTION-V5-20260731.json"
)
SELECTION_SHA256 = "f61509834d692d64ee17dad6030f3672969788aac39d3cb3d56b7d5db9b7207b"
CURRENT_EVIDENCE_RELATIVE = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-CURRENT-EVIDENCE-SELECTION-V3-20260731.json"
)
CURRENT_EVIDENCE_SHA256 = (
    "6518f7e876ce1c233fc055d3d20ad213088d361d784afbe5aa16d4f165e744f7"
)
COMPLETION_RELATIVE = Path("build/local-model-runs/run31n1c/completion_manifest.json")
COMPLETION_SHA256 = "fbc8278e9ae51be2d2bb54387b939540c7da97577025026440d4eebaecf0202c"
EXECUTION_RECEIPT_RELATIVE = Path(
    "build/workspace-local-runs/mdl31n1c/execution-receipt.json"
)
EXECUTION_RECEIPT_SHA256 = (
    "409132c223419423d6940bbde8b474a73e36be1e324b40827a4595542ca11a80"
)
OUTPUT_RELATIVE = Path(
    "build/ch-lt-structural-prediction-commitments/"
    "structural-dry-run-20260731T072027Z-v7-quant-contract"
)
SELECTED_COMMITMENT_SHA256 = (
    "67cc6fe722586cd4a02149252d1af7fbbb35e90a89e8bc7965f8d0bf87eda9b4"
)
SELECTED_PREDICTIONS_SHA256 = (
    "859c4622eb5b699f22242af0121141729d6c00472b6b9070bdb87c2ba9132302"
)
_PFC_ROLES = ("pfc_slow", "pfc_central", "pfc_fast")
_PFC_SHA256 = {
    "pfc_slow": "3acae6095b2536615b8e414dd11280c0a6566c542d872645011f28b2411696fb",
    "pfc_central": "f40b2f980b1788d4224f582a3bb97041ed6d22e2830dd5fe1645f6dc7e45ee62",
    "pfc_fast": "a91c8a7d3841f1a919f055d8d39d231eac5e163752b2306f2bfedac971b6ceae",
}
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_PFC_BYTES = 64 * 1024 * 1024
_LOCAL_TIMEZONE = "Europe/Zurich"
_HOURLY_SHAPE_FUNCTIONAL_CONTRACT: dict[str, object] = {
    "schema_version": "ch_lt_hourly_shape_functional_contract.v2",
    "status": "PRECOMMITTED_OUTCOME_BLIND_HOURLY_EVALUATION_ONLY",
    "truth_requirement": "NATIVE_CH_DAY_AHEAD_HOURLY_PRICES_WITH_GOVERNED_AVAILABLE_AT",
    "truth_timezone": _LOCAL_TIMEZONE,
    "quarter_hour_truth_required": False,
    "prediction_hourly_aggregation": "ARITHMETIC_MEAN_OF_FOUR_CONTIGUOUS_UTC_QUARTER_HOURS",
    "local_hour_windows": {
        "midday": {"start_inclusive": 11, "end_exclusive": 15},
        "evening": {"start_inclusive": 18, "end_exclusive": 22},
        "overnight": {"start_inclusive": 0, "end_exclusive": 6},
    },
    "functionals": [
        "hourly_mean_eur_mwh",
        "midday_mean_eur_mwh",
        "midday_premium_to_monthly_mean_eur_mwh",
        "evening_mean_eur_mwh",
        "evening_premium_to_monthly_mean_eur_mwh",
        "evening_minus_midday_eur_mwh",
        "overnight_mean_eur_mwh",
        "overnight_premium_to_monthly_mean_eur_mwh",
        "hourly_standard_deviation_eur_mwh_population",
        "hourly_p05_eur_mwh_linear",
        "hourly_p50_eur_mwh_linear",
        "hourly_p95_eur_mwh_linear",
        "hourly_min_eur_mwh",
        "hourly_max_eur_mwh",
        "mean_absolute_hourly_ramp_eur_mwh_utc",
        "negative_hour_count",
        "negative_hour_fraction",
        "negative_price_deficit_mean_all_hours_eur_mwh",
    ],
    "ex_post_scores": {
        "level": ["monthly_mean_error_eur_mwh"],
        "zero_mean_hourly_shape_primary": [
            "mae_eur_mwh",
            "rmse_eur_mwh",
            "pearson_correlation",
        ],
        "centering_rule": {
            "prediction": "PREDICTION_MINUS_ITS_OWN_MONTHLY_HOURLY_MEAN",
            "truth": "TRUTH_MINUS_ITS_OWN_MONTHLY_HOURLY_MEAN",
            "residual_sign": "PREDICTION_MINUS_TRUTH",
            "floating_point_comparison_absolute_tolerance": "1e-10",
        },
        "shape_functionals_primary": [
            "midday_premium_error_eur_mwh",
            "evening_premium_error_eur_mwh",
            "evening_minus_midday_error_eur_mwh",
        ],
        "absolute_price_tail_primary": [
            "absolute_negative_hour_count_error",
            "absolute_negative_hour_fraction_error",
            "absolute_negative_price_deficit_mean_error_eur_mwh",
            "absolute_minimum_price_error_eur_mwh",
        ],
        "absolute_error_aggregation_required": True,
        "raw_window_mean_errors_are_level_contaminated_diagnostics_only": True,
    },
    "primary_scenario": "central",
    "sensitivity_scenarios": ["slow", "fast"],
    "scenario_pooling_for_quality_claims": "FORBIDDEN",
    "same_origin_same_mask_baseline_required_for_quality_claim": True,
    "scenario_interpretation": "STRUCTURAL_SENSITIVITY_NOT_PROBABILISTIC_COVERAGE",
    "dst_rule": "AGGREGATE_IN_UTC_THEN_CLASSIFY_HOURS_IN_EUROPE_ZURICH",
    "market_resolution_rule": {
        "hourly_common_grid": "SUPPORTED_WITH_NATIVE_HOURLY_TRUTH",
        "quarter_hour_scores": "FORBIDDEN_UNTIL_CH_TRANSITION_IS_INDEPENDENTLY_ADMITTED",
        "transition_crossing_month": "UNSUPPORTED_IN_RESOLUTION_SPECIFIC_SCORE_GRAPH",
    },
}


class StructuralPredictionCommitmentError(ValueError):
    """Raised when the non-countable commitment inputs are not exact."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_bytes(value: object) -> bytes:
    return _canonical(value) + b"\n"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StructuralPredictionCommitmentError(f"{label} must be a mapping")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            (os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root)))
        ) == os.path.normcase(os.path.abspath(root))
    except ValueError:
        return False


def _read_exact(
    root: Path,
    relative: Path,
    expected_sha256: str,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    selected = root / relative
    payload = read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise StructuralPredictionCommitmentError(f"{label} SHA-256 mismatch")
    return payload


def _read_json(
    root: Path, relative: Path, expected_sha256: str, *, label: str
) -> Mapping[str, object]:
    payload = _read_exact(
        root,
        relative,
        expected_sha256,
        label=label,
        max_bytes=_MAX_JSON_BYTES,
    )
    try:
        value = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StructuralPredictionCommitmentError(f"{label} is not strict JSON") from exc
    return _mapping(value, label=label)


def _bound_pfc(
    root: Path, completion: Mapping[str, object], role: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    outputs = _mapping(completion.get("outputs"), label="completion outputs")
    record = _mapping(outputs.get(role), label=role)
    selected = Path(str(record.get("path")))
    if not selected.is_absolute():
        selected = root / selected
    selected = Path(os.path.abspath(selected))
    if not _inside(selected, root):
        raise StructuralPredictionCommitmentError(f"{role} escapes the repository")
    expected_sha256 = _PFC_SHA256[role]
    if record.get("sha256") != expected_sha256:
        raise StructuralPredictionCommitmentError(f"{role} completion binding mismatch")
    payload = read_stable_single_link_file(
        selected, label=role, max_bytes=_MAX_PFC_BYTES
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise StructuralPredictionCommitmentError(f"{role} SHA-256 mismatch")
    frame = pd.read_parquet(io.BytesIO(payload))
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise StructuralPredictionCommitmentError(f"{role} index is not timezone-aware")
    if "price_shape" not in frame.columns:
        raise StructuralPredictionCommitmentError(f"{role} price_shape is missing")
    return frame, {
        "path": selected.relative_to(root).as_posix(),
        "sha256": expected_sha256,
        "size_bytes": len(payload),
    }


def _finite_text(value: float, *, label: str) -> str:
    if not math.isfinite(value):
        raise StructuralPredictionCommitmentError(f"{label} is not finite")
    return format(value, ".17g")


def _hourly_shape_functionals(
    selected: pd.Series,
    *,
    expected_hour_count: int,
    label: str,
) -> dict[str, object]:
    utc = selected.copy()
    utc.index = utc.index.tz_convert("UTC")
    expected_index = pd.date_range(
        start=utc.index[0],
        periods=expected_hour_count * 4,
        freq="15min",
        tz="UTC",
    )
    if (
        not utc.index.is_unique
        or not utc.index.is_monotonic_increasing
        or not utc.index.equals(expected_index)
        or utc.index[0].minute != 0
        or utc.index[0].second != 0
        or utc.index[0].microsecond != 0
        or utc.index[0].nanosecond != 0
    ):
        raise StructuralPredictionCommitmentError(
            f"{label} quarter-hour grid is not exact and contiguous"
        )
    hour_key = utc.index.floor("h")
    counts = utc.groupby(hour_key, sort=True).count()
    hourly = utc.groupby(hour_key, sort=True).mean()
    if (
        len(hourly) != expected_hour_count
        or not (counts == 4).all()
        or not hourly.notna().all()
        or not hourly.index.is_monotonic_increasing
        or not hourly.index.is_unique
    ):
        raise StructuralPredictionCommitmentError(
            f"{label} cannot form exact native-hourly predictions"
        )
    local_hours = hourly.index.tz_convert(_LOCAL_TIMEZONE).hour
    midday = hourly[(local_hours >= 11) & (local_hours < 15)]
    evening = hourly[(local_hours >= 18) & (local_hours < 22)]
    overnight = hourly[(local_hours >= 0) & (local_hours < 6)]
    if midday.empty or evening.empty or overnight.empty:
        raise StructuralPredictionCommitmentError(f"{label} local-hour windows are empty")
    midday_mean = float(midday.mean())
    evening_mean = float(evening.mean())
    overnight_mean = float(overnight.mean())
    hourly_mean = float(hourly.mean())
    ramps = hourly.diff().dropna().abs()
    if ramps.empty:
        raise StructuralPredictionCommitmentError(f"{label} hourly ramps are empty")
    negative_count = int((hourly < 0.0).sum())
    return {
        "hourly_interval_count": len(hourly),
        "hourly_mean_eur_mwh": _finite_text(hourly_mean, label=f"{label} mean"),
        "midday_hour_count": len(midday),
        "midday_mean_eur_mwh": _finite_text(midday_mean, label=f"{label} midday"),
        "midday_premium_to_monthly_mean_eur_mwh": _finite_text(
            midday_mean - hourly_mean, label=f"{label} midday premium"
        ),
        "evening_hour_count": len(evening),
        "evening_mean_eur_mwh": _finite_text(evening_mean, label=f"{label} evening"),
        "evening_premium_to_monthly_mean_eur_mwh": _finite_text(
            evening_mean - hourly_mean, label=f"{label} evening premium"
        ),
        "evening_minus_midday_eur_mwh": _finite_text(
            evening_mean - midday_mean,
            label=f"{label} evening minus midday",
        ),
        "overnight_hour_count": len(overnight),
        "overnight_mean_eur_mwh": _finite_text(
            overnight_mean, label=f"{label} overnight"
        ),
        "overnight_premium_to_monthly_mean_eur_mwh": _finite_text(
            overnight_mean - hourly_mean, label=f"{label} overnight premium"
        ),
        "hourly_standard_deviation_eur_mwh_population": _finite_text(
            float(hourly.std(ddof=0)), label=f"{label} standard deviation"
        ),
        "hourly_p05_eur_mwh_linear": _finite_text(
            float(hourly.quantile(0.05, interpolation="linear")), label=f"{label} p05"
        ),
        "hourly_p50_eur_mwh_linear": _finite_text(
            float(hourly.quantile(0.50, interpolation="linear")), label=f"{label} p50"
        ),
        "hourly_p95_eur_mwh_linear": _finite_text(
            float(hourly.quantile(0.95, interpolation="linear")), label=f"{label} p95"
        ),
        "hourly_min_eur_mwh": _finite_text(float(hourly.min()), label=f"{label} minimum"),
        "hourly_max_eur_mwh": _finite_text(float(hourly.max()), label=f"{label} maximum"),
        "mean_absolute_hourly_ramp_eur_mwh_utc": _finite_text(
            float(ramps.mean()), label=f"{label} mean absolute ramp"
        ),
        "negative_hour_count": negative_count,
        "negative_hour_fraction": _finite_text(
            negative_count / len(hourly), label=f"{label} negative fraction"
        ),
        "negative_price_deficit_mean_all_hours_eur_mwh": _finite_text(
            float((-hourly.clip(upper=0.0)).mean()),
            label=f"{label} negative price deficit mean",
        ),
    }


def _selected_inventory_relative(root: Path, inventory_path: Path) -> Path:
    selected = inventory_path if inventory_path.is_absolute() else root / inventory_path
    selected = Path(os.path.abspath(selected))
    expected_parent = root / "build" / "ch-lt-origin-target-mask-inventories"
    if (
        not _inside(selected, expected_parent)
        or selected.parent != expected_parent
        or selected.suffix != ".json"
    ):
        raise StructuralPredictionCommitmentError(
            "inventory must be a direct JSON child of the governed inventory namespace"
        )
    return selected.relative_to(root)


def _publish_resumable_directory(
    output_directory: Path,
    *,
    prediction_payload: bytes,
    commitment_payload: bytes,
) -> None:
    """Seal both exact artifacts through a deterministic same-parent staging dir."""

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.parent / f".{output_directory.name}.staging"
    if staging.exists():
        if not staging.is_dir():
            raise StructuralPredictionCommitmentError(
                "staging residue is not a directory and requires repair"
            )
        entries = {entry.name for entry in staging.iterdir()}
        if not entries.issubset({"predictions.json", "commitment.json"}):
            raise StructuralPredictionCommitmentError(
                "staging residue contains unexpected entries and requires repair"
            )
    else:
        staging.mkdir()

    expected = {
        "predictions.json": prediction_payload,
        "commitment.json": commitment_payload,
    }
    for name, payload in expected.items():
        selected = staging / name
        if selected.exists():
            observed = read_stable_single_link_file(
                selected, label=f"staged {name}", max_bytes=_MAX_JSON_BYTES
            )
            if observed != payload:
                raise StructuralPredictionCommitmentError(
                    f"staged {name} is divergent and requires repair"
                )
        else:
            write_durable_exact_bytes(selected, payload, label=f"staged {name}")

    for name, payload in expected.items():
        observed = read_stable_single_link_file(
            staging / name, label=f"staged {name}", max_bytes=_MAX_JSON_BYTES
        )
        if observed != payload:
            raise StructuralPredictionCommitmentError(f"staged {name} changed before seal")
    try:
        os.replace(staging, output_directory)
    except FileExistsError as exc:
        raise StructuralPredictionCommitmentError(
            "output appeared concurrently; publication not selected"
        ) from exc


def build_commitment(
    root: Path,
    output_directory: Path,
    *,
    inventory_path: Path,
    expected_inventory_sha256: str,
) -> dict[str, object]:
    root = assert_absolute_path_has_no_links(root)
    if root != ROOT or Path.cwd() != root:
        raise StructuralPredictionCommitmentError(
            "cwd and root must be the canonical repository"
        )
    output_directory = Path(os.path.abspath(output_directory))
    if not _inside(output_directory, root / "build"):
        raise StructuralPredictionCommitmentError("output must remain below build/")
    if output_directory.exists():
        raise StructuralPredictionCommitmentError("output directory already exists")

    if (
        len(expected_inventory_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_inventory_sha256)
    ):
        raise StructuralPredictionCommitmentError(
            "expected inventory SHA-256 must be lowercase hexadecimal"
        )
    inventory_relative = _selected_inventory_relative(root, inventory_path)
    if (
        inventory_relative != SELECTED_INVENTORY_RELATIVE
        or expected_inventory_sha256 != SELECTED_INVENTORY_SHA256
    ):
        raise StructuralPredictionCommitmentError(
            "inventory is not the selected outcome-blind origin inventory"
        )

    inventory_payload = _read_exact(
        root,
        inventory_relative,
        expected_inventory_sha256,
        label="origin target inventory",
        max_bytes=_MAX_JSON_BYTES,
    )
    try:
        inventory_value = load_strict_json(inventory_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StructuralPredictionCommitmentError(
            "origin target inventory is not strict JSON"
        ) from exc
    inventory = _mapping(inventory_value, label="origin target inventory")
    try:
        inventory_assessment = origin_inventory.assess_structural_inventory(
            inventory,
            document_bytes=inventory_payload,
        )
    except ValueError as exc:
        raise StructuralPredictionCommitmentError(
            "origin target inventory semantics are invalid"
        ) from exc
    if (
        inventory_assessment.get("inventory_id") != inventory.get("inventory_id")
        or inventory_assessment.get("outcome_blind") is not True
        or inventory_assessment.get("countable_prospective_origin") is not False
    ):
        raise StructuralPredictionCommitmentError(
            "origin target inventory assessment is not outcome-blind"
        )
    candidate = _read_json(
        root, CANDIDATE_RELATIVE, CANDIDATE_SHA256, label="candidate core"
    )
    selection = _read_json(
        root, SELECTION_RELATIVE, SELECTION_SHA256, label="local quality selection"
    )
    current_evidence = _read_json(
        root,
        CURRENT_EVIDENCE_RELATIVE,
        CURRENT_EVIDENCE_SHA256,
        label="current evidence selection",
    )
    completion = _read_json(
        root, COMPLETION_RELATIVE, COMPLETION_SHA256, label="model completion"
    )
    execution = _read_json(
        root,
        EXECUTION_RECEIPT_RELATIVE,
        EXECUTION_RECEIPT_SHA256,
        label="model execution receipt",
    )

    origin = _mapping(inventory.get("origin"), label="inventory origin")
    targets = inventory.get("targets")
    if (
        inventory.get("schema_version") != "ch_lt_origin_target_mask_inventory.v2"
        or inventory.get("predictions_sealed") is not False
        or inventory.get("outcomes_opened") is not False
        or inventory.get("future_holdout_consumed") is not False
        or inventory.get("t057_consumed") is not False
        or not isinstance(targets, list)
        or len(targets) != 36
        or [item.get("lead_month") for item in targets] != list(range(1, 37))
    ):
        raise StructuralPredictionCommitmentError("inventory is not an outcome-blind M01-M36 grid")
    if (
        candidate.get("schema_version") != "ch_lt_successor_candidate_core.v3"
        or candidate.get("scientific_admission") is not False
        or candidate.get("future_holdout_consumed") is not False
        or candidate.get("t057_consumed") is not False
        or selection.get("status")
        != "CURRENT_LOCAL_ENGINEERING_QUALITY_SELECTION_NO_GO"
        or selection.get("scientific_admission") is not False
        or selection.get("production_authorization") is not False
        or current_evidence.get("schema_version")
        != "ch_lt_current_evidence_selection.v3"
        or current_evidence.get("status")
        != "CURRENT_LOCAL_DIAGNOSTIC_SELECTION_ONLY_NOT_AUTHORITY"
        or current_evidence.get("independent_rolling_origin_count") != 0
        or current_evidence.get("future_holdout_consumed") is not False
        or current_evidence.get("t057_consumed") is not False
        or current_evidence.get("scientific_admission") is not False
        or current_evidence.get("production_authorization") is not False
        or execution.get("status") != "TARGET_EXIT_ZERO_NOT_AUTHORITY"
        or execution.get("scientific_authorization") is not False
        or completion.get("forward_source_kind") != "TEST_FIXTURE"
        or completion.get("monthly_level_authority") != "solver"
    ):
        raise StructuralPredictionCommitmentError("selected model chain is not local NO_GO")

    frames: dict[str, pd.DataFrame] = {}
    pfc_bindings: dict[str, object] = {}
    for role in _PFC_ROLES:
        frames[role], pfc_bindings[role] = _bound_pfc(root, completion, role)

    predictions: list[dict[str, object]] = []
    universe: list[dict[str, object]] = []
    for raw_target in targets:
        target = _mapping(raw_target, label="inventory target")
        start = pd.Timestamp(str(target.get("delivery_start_utc")))
        end = pd.Timestamp(str(target.get("delivery_end_utc")))
        expected_count = int(target.get("expected_quarter_hour_interval_count"))
        expected_hour_count = int(target.get("expected_native_hourly_interval_count"))
        universe.append(
            {
                "target_id": target.get("target_id"),
                "lead_month": target.get("lead_month"),
                "delivery_month": target.get("delivery_month"),
                "delivery_start_utc": target.get("delivery_start_utc"),
                "delivery_end_utc": target.get("delivery_end_utc"),
                "expected_quarter_hour_interval_count": expected_count,
                "expected_native_hourly_interval_count": target.get(
                    "expected_native_hourly_interval_count"
                ),
            }
        )
        for role in _PFC_ROLES:
            frame = frames[role]
            selected = frame.loc[(frame.index >= start) & (frame.index < end), "price_shape"]
            if len(selected) != expected_count or not selected.notna().all():
                raise StructuralPredictionCommitmentError(
                    f"{role} target {target.get('target_id')} is incomplete"
                )
            functionals = _hourly_shape_functionals(
                selected,
                expected_hour_count=expected_hour_count,
                label=f"{role} target {target.get('target_id')}",
            )
            predictions.append(
                {
                    "target_id": target.get("target_id"),
                    "lead_month": target.get("lead_month"),
                    "delivery_month": target.get("delivery_month"),
                    "scenario": role.removeprefix("pfc_"),
                    "quarter_hour_interval_count": len(selected),
                    "native_hourly_interval_count": expected_hour_count,
                    "monthly_mean_eur_mwh": format(float(selected.mean()), ".17g"),
                    "hourly_shape_functionals": functionals,
                    "probabilistic_interval_supported": False,
                }
            )

    prediction_core: dict[str, object] = {
        "schema_version": "ch_lt_structural_prediction_set.v3",
        "status": "SEALED_LOCAL_STRUCTURAL_DRY_RUN_NONCOUNTABLE_NO_GO",
        "market": "CH",
        "origin_as_of_utc": origin.get("as_of_utc"),
        "origin_id": origin.get("origin_id"),
        "inventory_id": inventory.get("inventory_id"),
        "target_count": 36,
        "prediction_count": 108,
        "scenario_order": ["slow", "central", "fast"],
        "primary_scenario": "central",
        "sensitivity_scenarios": ["slow", "fast"],
        "scenario_pooling_for_quality_claims": "FORBIDDEN",
        "hourly_shape_functional_contract": _HOURLY_SHAPE_FUNCTIONAL_CONTRACT,
        "predictions": predictions,
        "outcomes_opened": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    prediction_set_id = hashlib.sha256(_canonical(prediction_core)).hexdigest()
    prediction_document = {**prediction_core, "prediction_set_id": prediction_set_id}
    prediction_payload = _artifact_bytes(prediction_document)

    scenario_payload = {
        "scenario_order": ["slow", "central", "fast"],
        "primary_scenario": "central",
        "sensitivity_scenarios": ["slow", "fast"],
        "scenario_pooling_for_quality_claims": "FORBIDDEN",
        "structural_weights": ["0.25", "0.50", "0.25"],
        "structural_weights_use": "DISPLAY_ONLY_NOT_SCORING_NOT_PROBABILITY",
        "probabilistic_interpretation": False,
        "hourly_shape_functional_contract": _HOURLY_SHAPE_FUNCTIONAL_CONTRACT,
        "pfc_bindings": pfc_bindings,
    }
    mask_policy = _mapping(inventory.get("mask_policy"), label="mask policy")
    commitments = {
        "prediction_commitment_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "scenario_commitment_sha256": hashlib.sha256(
            _canonical(scenario_payload)
        ).hexdigest(),
        "structural_target_universe_commitment_sha256": hashlib.sha256(
            _canonical(universe)
        ).hexdigest(),
        "ex_ante_evaluation_mask_rule_commitment_sha256": hashlib.sha256(
            _canonical(mask_policy)
        ).hexdigest(),
        "candidate_hypothesis_and_procedure_commitment_sha256": CANDIDATE_SHA256,
        "hourly_shape_functional_contract_sha256": hashlib.sha256(
            _canonical(_HOURLY_SHAPE_FUNCTIONAL_CONTRACT)
        ).hexdigest(),
    }
    commitment_core: dict[str, object] = {
        "schema_version": "ch_lt_structural_prediction_commitment.v3",
        "status": "SEALED_LOCAL_STRUCTURAL_DRY_RUN_NONCOUNTABLE_NO_GO",
        "market": "CH",
        "origin_as_of_utc": origin.get("as_of_utc"),
        "origin_id": origin.get("origin_id"),
        "target_count": 36,
        "prediction_count": 108,
        "monthly_level_authority": "solver",
        "timing_boundary": {
            "declared_data_cutoff_utc_untrusted": origin.get("as_of_utc"),
            "prediction_sealed_at_utc_untrusted": None,
            "registry_linearized_at_utc": None,
            "first_target_delivery_start_utc": universe[0]["delivery_start_utc"],
            "countable_origin_requires_all_three_times_before_first_delivery": True,
        },
        "quality_claim_boundary": {
            "primary_scenario": "central",
            "sensitivity_scenarios": ["slow", "fast"],
            "same_origin_same_mask_baseline_attached": False,
            "quality_or_noninferiority_claim_authorized": False,
        },
        "bindings": {
            "origin_target_mask_inventory": {
                "path": inventory_relative.as_posix(),
                "sha256": expected_inventory_sha256,
                "inventory_id": inventory.get("inventory_id"),
            },
            "candidate_core": {
                "path": CANDIDATE_RELATIVE.as_posix(),
                "sha256": CANDIDATE_SHA256,
                "core_id": candidate.get("core_id"),
            },
            "local_quality_selection": {
                "path": SELECTION_RELATIVE.as_posix(),
                "sha256": SELECTION_SHA256,
                "selection_id": selection.get("selection_id"),
            },
            "current_evidence_selection": {
                "path": CURRENT_EVIDENCE_RELATIVE.as_posix(),
                "sha256": CURRENT_EVIDENCE_SHA256,
                "registry_id": current_evidence.get("registry_id"),
                "independent_rolling_origin_count": current_evidence.get(
                    "independent_rolling_origin_count"
                ),
            },
            "model_completion": {
                "path": COMPLETION_RELATIVE.as_posix(),
                "sha256": COMPLETION_SHA256,
            },
            "model_execution_receipt": {
                "path": EXECUTION_RECEIPT_RELATIVE.as_posix(),
                "sha256": EXECUTION_RECEIPT_SHA256,
            },
            "pfc_outputs": pfc_bindings,
        },
        "sealed_payload": {
            "path": "predictions.json",
            "sha256": hashlib.sha256(prediction_payload).hexdigest(),
            "prediction_set_id": prediction_set_id,
        },
        "commitments": commitments,
        "authority_boundary": {
            "local_clock_trusted": False,
            "independent_trusted_time_receipt_present": False,
            "independent_signature_present": False,
            "builder_inaccessible_external_cas_receipt_present": False,
            "fresh_external_head_observation_present": False,
            "countable_prospective_origin": False,
            "training_eligible": False,
            "model_selection_eligible": False,
            "truth_open_authorized": False,
            "outcomes_opened": False,
            "future_holdout_consumed": False,
            "t057_consumed": False,
            "scientific_admission": False,
            "production_authorization": False,
            "promotion_gate": False,
        },
        "limitations": [
            "TEST_FIXTURE_MONTHLY_LEVEL_NOT_FRESH_GOVERNED_CH_EEX",
            "DE_LU_INTRADAY_PROXY_NOT_DIRECT_CH_SHAPE_AUTHORITY",
            "EMPTY_P10_P90_NOT_PROBABILISTICALLY_CALIBRATED",
            "NO_INDEPENDENT_TRUSTED_TIME_SIGNATURE_OR_EXTERNAL_CAS",
            "STRUCTURAL_DRY_RUN_CANNOT_BE_COUNTED_AS_PROSPECTIVE_EVIDENCE",
            "CURRENT_LOCAL_CAPTURE_LEDGER_DOES_NOT_REQUALIFY_THE_FROZEN_MODEL",
            "SAME_ORIGIN_SAME_MASK_BASELINE_NOT_ATTACHED",
            "PREDICTION_SEAL_TIME_AND_REGISTRY_LINEARIZATION_NOT_BOUND_IN_THIS_PAYLOAD",
        ],
    }
    commitment_id = hashlib.sha256(_canonical(commitment_core)).hexdigest()
    commitment_document = {**commitment_core, "commitment_id": commitment_id}
    commitment_payload = _artifact_bytes(commitment_document)

    _publish_resumable_directory(
        output_directory,
        prediction_payload=prediction_payload,
        commitment_payload=commitment_payload,
    )
    return commitment_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = build_commitment(
        ROOT,
        args.output_directory,
        inventory_path=args.inventory,
        expected_inventory_sha256=args.expected_inventory_sha256,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
