"""Offline lambda calibration harness for the monthly forward curve solver.

The harness is intentionally research-only.  It evaluates candidate monthly
curve configurations on rolling-origin snapshots by withholding monthly or
quarterly products, solving from a degraded quote set, and scoring the solved
curve against the withheld same-snapshot target.  All solver inputs and priors
are built from the visible quote set plus history strictly before the origin
date.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pfc_shaping.calibration.monthly_curve_audit import audit_monthly_curve_shape
from pfc_shaping.calibration.monthly_curve_config_identity import (
    _sha256_text,
    canonical_json,
    config_hash,
    lambda_grid_hash,
)
from pfc_shaping.calibration.monthly_curve_config_identity import (
    config_payload as _config_payload,
)
from pfc_shaping.calibration.monthly_curve_priors import (
    MonthlyShapePrior,
    build_fused_shape_prior,
    build_history_shape_prior,
    build_neighbor_panel_shape_prior,
    build_structural_monthly_shape_prior_from_history,
)
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    MonthlyCurveConfig,
    build_monthly_constraint_system,
    product_periods,
    solve_monthly_forward_curve_from_constraints,
)
from pfc_shaping.data.eex_historical_vintage import (
    EexHistoricalVintageError,
    VerifiedEexHistoricalVintageCatalog,
    validate_eex_historical_vintage_frame,
    verify_eex_historical_vintage_catalog,
)

SCORING_COLUMNS = [
    "config_hash",
    "origin_date",
    "snapshot_id",
    "source_file_hash",
    "market",
    "withheld_product",
    "withheld_load_type",
    "withheld_tenor",
    "withheld_horizon_years",
    "withheld_horizon_bucket",
    "product_start",
    "product_end",
    "target_price",
    "predicted_price",
    "abs_error",
    "signed_error",
    "constraint_residual_max",
    "curvature_score",
    "same_month_rank_score",
    "historical_outlier_score",
    "neighbor_disagreement_score",
    "unsupported_gate_count",
    "critical_gate_count",
    "status",
    "unsupported_reason",
    "sample_size",
]

FINAL_STATUSES = {
    "PASS_IMPLEMENTATION_ONLY",
    "PASS_CALIBRATION_CANDIDATE_NOT_PRODUCTION_APPROVED",
    "UNSUPPORTED_INSUFFICIENT_HISTORY",
    "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS",
    "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA",
    "FAIL_LEAKAGE_DETECTED",
    "FAIL_HARD_CONSTRAINT_VIOLATION",
    "FAIL_NON_REPRODUCIBLE_HASH",
    "FAIL_ARTIFACT_INCOMPLETE",
}

FAIL_STATUSES = {
    "FAIL_LEAKAGE_DETECTED",
    "FAIL_HARD_CONSTRAINT_VIOLATION",
    "FAIL_NON_REPRODUCIBLE_HASH",
    "FAIL_ARTIFACT_INCOMPLETE",
}

_MONTH_RE = r"^\d{4}-\d{2}$"
_QUARTER_RE = r"^\d{4}-Q[1-4]$"
_YEAR_RE = r"^\d{4}$"
GOVERNED_CALIBRATION_PROFILE = "tier2_governed_minimums.v1"
_VERIFIED_CALIBRATION_ARTIFACT_TOKEN = object()


@dataclass(frozen=True)
class LambdaCalibrationSettings:
    market: str = "CH"
    load_type: str = "BASE"
    timezone: str = "Europe/Zurich"
    neighbor_markets: tuple[str, ...] = ("DE", "FR", "AT", "IT")
    min_valid_origins: int = 3
    min_withheld_monthly: int = 3
    min_withheld_quarterly: int = 2
    min_history_snapshots: int = 24
    max_withheld_per_origin: int = 4
    quote_consistency_tolerance: float = 0.01
    hard_constraint_tolerance: float = 1e-8
    history_lookback_years: int | None = 6
    structural_weight: float = 1.0
    panel_weight: float = 1.0
    history_weight: float = 0.5
    allow_template_structural_fallback: bool = True
    structural_amplitude_eur_mwh: float = 110.0
    min_structural_snapshots: int = 24
    identifiability_min_abs_error_improvement: float = 0.05
    identifiability_min_rel_error_improvement: float = 0.01


_GOVERNED_MINIMUM_SETTINGS = {
    "min_valid_origins": LambdaCalibrationSettings().min_valid_origins,
    "min_withheld_monthly": LambdaCalibrationSettings().min_withheld_monthly,
    "min_withheld_quarterly": LambdaCalibrationSettings().min_withheld_quarterly,
    "min_history_snapshots": LambdaCalibrationSettings().min_history_snapshots,
    "min_structural_snapshots": LambdaCalibrationSettings().min_structural_snapshots,
    "max_withheld_per_origin": LambdaCalibrationSettings().max_withheld_per_origin,
    "identifiability_min_abs_error_improvement": (
        LambdaCalibrationSettings().identifiability_min_abs_error_improvement
    ),
    "identifiability_min_rel_error_improvement": (
        LambdaCalibrationSettings().identifiability_min_rel_error_improvement
    ),
}
_GOVERNED_MAXIMUM_SETTINGS = {
    "quote_consistency_tolerance": LambdaCalibrationSettings().quote_consistency_tolerance,
    "hard_constraint_tolerance": LambdaCalibrationSettings().hard_constraint_tolerance,
}
_GOVERNED_EXACT_SETTINGS = {
    name: value
    for name, value in LambdaCalibrationSettings().__dict__.items()
    if name not in _GOVERNED_MINIMUM_SETTINGS
    and name not in _GOVERNED_MAXIMUM_SETTINGS
}


@dataclass(frozen=True)
class WithheldProduct:
    market: str
    product: str
    load_type: str
    price: float
    origin_date: pd.Timestamp
    source: str = ""
    snapshot_id: str = ""

    @property
    def tenor(self) -> str:
        return product_tenor(self.product)

    @property
    def key(self) -> str:
        return quote_key(self.market, self.load_type, self.product)


@dataclass(frozen=True)
class LambdaCandidateConfig:
    monthly_config: MonthlyCurveConfig
    history_lookback_years: int | None


@dataclass(frozen=True)
class CalibrationArtifacts:
    scoring: pd.DataFrame
    manifest: dict[str, object]
    summary: dict[str, object]
    candidate_config: dict[str, object] | None
    _verification_token: InitVar[object | None] = None
    verified: bool = field(init=False)
    sealed_content_sha256: str = field(init=False)

    def __post_init__(self, _verification_token: object | None) -> None:
        object.__setattr__(
            self,
            "verified",
            _verification_token is _VERIFIED_CALIBRATION_ARTIFACT_TOKEN,
        )
        object.__setattr__(
            self,
            "sealed_content_sha256",
            _calibration_artifact_content_sha256(
                self.scoring,
                self.manifest,
                self.summary,
                self.candidate_config,
            )
            if _verification_token is _VERIFIED_CALIBRATION_ARTIFACT_TOKEN
            else "",
        )


def run_verified_lambda_calibration(
    history_path: str | Path,
    catalog_path: str | Path,
    *,
    grid: Mapping[str, object],
    settings: LambdaCalibrationSettings | None = None,
    max_origins: int | None = None,
    max_configs: int | None = None,
    smoke: bool = False,
    command_line: Sequence[str] | None = None,
) -> CalibrationArtifacts:
    """Run calibration only through exact signed catalog and Parquet bytes."""

    if not smoke and (max_origins is not None or max_configs is not None):
        raise EexHistoricalVintageError(
            "max_origins/max_configs are smoke-only and cannot reduce governed calibration"
        )
    selected_history = Path(history_path).resolve()
    selected_catalog = Path(catalog_path).resolve()
    try:
        catalog = json.loads(selected_catalog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EexHistoricalVintageError("EEX vintage catalog is unreadable") from exc
    history, evidence = verify_eex_historical_vintage_catalog(
        catalog,
        catalog_path=selected_catalog,
        history_path=selected_history,
    )
    effective_settings = _settings_from_grid(grid, settings=settings, smoke=smoke)
    governance_profile = "VERIFIED_VINTAGE_SMOKE_DIAGNOSTIC_ONLY"
    execution_environment = None
    if not smoke:
        _validate_governed_calibration_settings(effective_settings)
        execution_environment = _execution_environment_receipt()
        if bool(execution_environment["git_worktree_dirty"]):
            raise EexHistoricalVintageError(
                "governed calibration requires a clean, committed source worktree"
            )
        governance_profile = GOVERNED_CALIBRATION_PROFILE
    return _run_lambda_calibration_core(
        history,
        grid=grid,
        settings=effective_settings,
        source_file_hash=evidence.history_sha256,
        max_origins=max_origins,
        max_configs=max_configs,
        smoke=smoke,
        command_line=command_line,
        input_parquet_path=str(selected_history),
        vintage_evidence=evidence,
        governance_profile=governance_profile,
        expected_execution_environment=execution_environment,
        artifact_verification_token=_VERIFIED_CALIBRATION_ARTIFACT_TOKEN,
    )


def _run_lambda_calibration_core(
    history: pd.DataFrame,
    *,
    grid: Mapping[str, object],
    settings: LambdaCalibrationSettings | None = None,
    source_file_hash: str = "",
    max_origins: int | None = None,
    max_configs: int | None = None,
    smoke: bool = False,
    command_line: Sequence[str] | None = None,
    input_parquet_path: str = "",
    vintage_evidence: VerifiedEexHistoricalVintageCatalog | None = None,
    allow_unverified_vintage_fixture: bool = False,
    governance_profile: str = "UNVERIFIED_INTERNAL_DIAGNOSTIC",
    expected_execution_environment: Mapping[str, object] | None = None,
    artifact_verification_token: object | None = None,
) -> CalibrationArtifacts:
    """Internal core; unverified fixtures are never a governed CLI surface."""

    settings = _settings_from_grid(grid, settings=settings, smoke=smoke)
    history = normalize_history(history)
    history = validate_eex_historical_vintage_frame(history)
    if vintage_evidence is None and not allow_unverified_vintage_fixture:
        raise EexHistoricalVintageError(
            "signed eex_historical_vintage_catalog.v1 evidence is required"
        )
    if vintage_evidence is not None:
        if vintage_evidence.verified is not True:
            raise EexHistoricalVintageError(
                "EEX vintage evidence was not created by the strict verifier"
            )
        if not source_file_hash or vintage_evidence.history_sha256 != source_file_hash:
            raise EexHistoricalVintageError(
                "verified EEX vintage catalog does not bind the consumed history bytes"
            )
        history.attrs["verified_vintage_catalog"] = {
            "catalog_id": vintage_evidence.catalog_id,
            "catalog_sha256": vintage_evidence.catalog_sha256,
            "history_sha256": vintage_evidence.history_sha256,
            "snapshot_count": vintage_evidence.snapshot_count,
            "source_document_count": vintage_evidence.source_document_count,
            "data_cutoff_utc": vintage_evidence.data_cutoff_utc,
            "status": "VERIFIED_SIGNED_IMMUTABLE_VINTAGES",
        }
    else:
        history.attrs["verified_vintage_catalog"] = {
            "status": "UNVERIFIED_TEST_FIXTURE",
        }
    configs = list(iter_candidate_configs(grid))
    if not configs:
        raise ValueError("lambda grid did not produce any candidate configuration")
    if smoke:
        configs = configs[: max_configs or 2]
        max_origins = max_origins or 2
    elif max_configs is not None:
        configs = configs[: int(max_configs)]

    origins = select_origins(history, settings=settings, max_origins=max_origins)
    scoring_rows: list[dict[str, object]] = []
    excluded_reasons: Counter[str] = Counter()
    withheld_counts: Counter[tuple[str, str, str]] = Counter()

    for origin in origins:
        full_quote_set = quote_set_for_origin(history, origin)
        withheld = select_withheld_products(
            full_quote_set,
            settings=settings,
            max_products=settings.max_withheld_per_origin if not smoke else min(2, settings.max_withheld_per_origin),
        )
        if not withheld:
            excluded_reasons["no_withheld_products"] += 1
            continue
        for withheld_product in withheld:
            withheld_counts[
                (
                    withheld_product.market,
                    withheld_product.load_type,
                    withheld_product.tenor,
                )
            ] += 1
            masked = mask_quote_sets(
                full_quote_set,
                withheld_product,
                neighbor_markets=settings.neighbor_markets,
            )
            try:
                validate_masked_inputs(masked, withheld_product)
            except ValueError:
                excluded_reasons["mask_leakage_detected"] += 1
                raise
            for cfg in configs:
                history_view = history_before_origin(
                    history,
                    origin,
                    lookback_years=_candidate_history_lookback(cfg, settings),
                )
                validate_history_point_in_time(history_view, origin)
                scoring_rows.append(
                    _score_one_case(
                        masked=masked,
                        withheld=withheld_product,
                        config=cfg,
                        settings=settings,
                        history_view=history_view,
                        source_file_hash=source_file_hash,
                    )
                )

    scoring = pd.DataFrame(scoring_rows, columns=SCORING_COLUMNS)
    summary = summarize_calibration(
        scoring,
        configs=configs,
        settings=settings,
        withheld_counts=withheld_counts,
        excluded_reasons=excluded_reasons,
    )
    if smoke and str(summary.get("final_status", "")).startswith(
        "PASS_CALIBRATION_CANDIDATE"
    ):
        summary["final_status"] = "PASS_IMPLEMENTATION_ONLY"
    if smoke:
        summary["selected_config_hash"] = None
        summary["selected_mae"] = None
        summary["selection_reason"] = (
            "smoke execution validates implementation only and cannot select a model"
        )
    execution_environment = _execution_environment_receipt()
    if expected_execution_environment is not None:
        if canonical_json(execution_environment) != canonical_json(
            expected_execution_environment
        ):
            raise EexHistoricalVintageError(
                "governed calibration execution environment changed during the run"
            )
        if bool(execution_environment["git_worktree_dirty"]):
            raise EexHistoricalVintageError(
                "governed calibration source worktree became dirty during the run"
            )
    manifest = build_calibration_manifest(
        history=history,
        scoring=scoring,
        summary=summary,
        settings=settings,
        grid=grid,
        source_file_hash=source_file_hash,
        command_line=command_line or (),
        input_parquet_path=input_parquet_path,
        withheld_counts=withheld_counts,
        excluded_reasons=excluded_reasons,
        governance_profile=governance_profile,
        execution_environment=execution_environment,
    )
    candidate = None
    if not smoke:
        candidate = build_candidate_config(
            scoring,
            summary=summary,
            configs=configs,
            settings=settings,
            grid_hash=lambda_grid_hash(grid),
            source_data_hash=source_file_hash,
            withheld_counts=withheld_counts,
            excluded_reasons=excluded_reasons,
        )
    _validate_artifacts(scoring, manifest, summary, candidate)
    return CalibrationArtifacts(
        scoring,
        manifest,
        summary,
        candidate,
        _verification_token=artifact_verification_token,
    )


def write_calibration_artifacts(artifacts: CalibrationArtifacts, output_dir: Path) -> None:
    if artifacts.verified is not True:
        raise EexHistoricalVintageError(
            "only run_verified_lambda_calibration artifacts may be published"
        )
    payloads = _calibration_artifact_payloads(
        artifacts.scoring,
        artifacts.manifest,
        artifacts.summary,
        artifacts.candidate_config,
    )
    current_digest = _serialized_calibration_payloads_sha256(payloads)
    if current_digest != artifacts.sealed_content_sha256:
        raise EexHistoricalVintageError(
            "verified calibration artifacts changed after strict evaluation"
        )
    _write_calibration_artifacts_core(
        artifacts,
        output_dir,
        serialized_payloads=payloads,
    )


def _write_calibration_artifacts_core(
    artifacts: CalibrationArtifacts,
    output_dir: Path,
    *,
    serialized_payloads: Mapping[str, bytes] | None = None,
) -> None:
    """Publish one immutable, complete artifact set to a new run directory."""

    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"calibration output directory already exists: {output_dir}")
    stage = output_dir.parent / f".{output_dir.name}-{uuid.uuid4().hex}.staging"
    stage.mkdir(exist_ok=False)
    payloads = dict(
        serialized_payloads
        or _calibration_artifact_payloads(
            artifacts.scoring,
            artifacts.manifest,
            artifacts.summary,
            artifacts.candidate_config,
        )
    )
    summary_payload = json.loads(payloads["calibration_summary.json"])
    inventory = {
        "schema_version": "monthly_curve_calibration_artifact_set.v1",
        "final_status": str(summary_payload.get("final_status", "")),
        "candidate_config_present": "candidate_config.yaml" in payloads,
        "files": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(payloads.items())
        },
    }
    payloads["artifact_set_manifest.json"] = json.dumps(
        inventory,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    published = False
    try:
        for name, payload in payloads.items():
            _write_fsync(stage / name, payload)
        stage.replace(output_dir)
        published = True
        verify_calibration_artifact_set(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if published and output_dir.exists():
            quarantine = output_dir.parent / (
                f".{output_dir.name}-{uuid.uuid4().hex}.invalid"
            )
            output_dir.replace(quarantine)
            shutil.rmtree(quarantine, ignore_errors=True)
        raise


def _calibration_artifact_payloads(
    scoring: pd.DataFrame,
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    candidate_config: Mapping[str, object] | None,
) -> dict[str, bytes]:
    payloads = {
        "scoring.csv": scoring.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        "calibration_manifest.json": json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            default=str,
        ).encode("utf-8"),
        "calibration_summary.json": json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            default=str,
        ).encode("utf-8"),
    }
    if candidate_config is not None:
        payloads["candidate_config.yaml"] = yaml.safe_dump(
            dict(candidate_config),
            sort_keys=True,
        ).encode("utf-8")
    return payloads


def _calibration_artifact_content_sha256(
    scoring: pd.DataFrame,
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    candidate_config: Mapping[str, object] | None,
) -> str:
    payloads = _calibration_artifact_payloads(
        scoring,
        manifest,
        summary,
        candidate_config,
    )
    return _serialized_calibration_payloads_sha256(payloads)


def _serialized_calibration_payloads_sha256(
    payloads: Mapping[str, bytes],
) -> str:
    receipts = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in sorted(payloads.items())
    }
    return hashlib.sha256(
        json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_calibration_artifact_set(output_dir: str | Path) -> dict[str, object]:
    """Replay the exact immutable file inventory for one calibration run."""

    root = Path(output_dir).resolve()
    manifest_path = root / "artifact_set_manifest.json"
    try:
        inventory = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("calibration artifact-set manifest is unreadable") from exc
    if inventory.get("schema_version") != "monthly_curve_calibration_artifact_set.v1":
        raise ValueError("calibration artifact-set schema mismatch")
    files = inventory.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("calibration artifact-set inventory is missing")
    final_status = str(inventory.get("final_status", ""))
    candidate_required = final_status.startswith("PASS_CALIBRATION_CANDIDATE")
    required_payload_names = {
        "scoring.csv",
        "calibration_manifest.json",
        "calibration_summary.json",
    }
    expected_payload_names = set(required_payload_names)
    if candidate_required:
        expected_payload_names.add("candidate_config.yaml")
    if {str(name) for name in files} != expected_payload_names:
        raise ValueError("calibration artifact-set required role mismatch")
    entries = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("calibration artifact-set contains non-regular entries")
    actual_names = {path.name for path in entries}
    expected_names = {str(name) for name in files} | {manifest_path.name}
    if actual_names != expected_names:
        raise ValueError("calibration artifact-set file inventory mismatch")
    for name, raw in files.items():
        if not isinstance(raw, Mapping):
            raise ValueError("calibration artifact-set receipt is invalid")
        payload = (root / str(name)).read_bytes()
        if raw.get("size_bytes") != len(payload) or str(raw.get("sha256", "")) != (
            hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError(f"calibration artifact-set receipt mismatch: {name}")
    candidate_present = "candidate_config.yaml" in files
    if inventory.get("candidate_config_present") is not candidate_present:
        raise ValueError("calibration candidate presence flag mismatch")
    try:
        summary = json.loads((root / "calibration_summary.json").read_text(encoding="utf-8"))
        run_manifest = json.loads(
            (root / "calibration_manifest.json").read_text(encoding="utf-8")
        )
        scoring = pd.read_csv(root / "scoring.csv")
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("calibration artifact-set semantic payload is unreadable") from exc
    if str(summary.get("final_status", "")) != final_status or str(
        run_manifest.get("final_status", "")
    ) != final_status:
        raise ValueError("calibration artifact-set final status parity mismatch")
    if bool(summary.get("production_approved")) or bool(
        run_manifest.get("production_approved")
    ):
        raise ValueError("calibration artifact-set cannot be production approved")
    if list(scoring.columns) != SCORING_COLUMNS:
        raise ValueError("calibration artifact-set scoring schema mismatch")
    if candidate_required:
        try:
            candidate = yaml.safe_load(
                (root / "candidate_config.yaml").read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("calibration candidate config is unreadable") from exc
        if not isinstance(candidate, Mapping) or candidate.get("selection_status") != "SELECTED":
            raise ValueError("calibration candidate selection contract mismatch")
        if bool(candidate.get("production_approved")):
            raise ValueError("calibration candidate cannot be production approved")
        expected_scope = {
            "market": "CH",
            "load_type": "BASE",
            "timezone": "Europe/Zurich",
            "profile": GOVERNED_CALIBRATION_PROFILE,
        }
        if candidate.get("governed_scope") != expected_scope:
            raise ValueError("calibration candidate governed scope mismatch")
        canonical_config = candidate.get("canonical_config")
        if not isinstance(canonical_config, Mapping) or str(
            candidate.get("config_hash", "")
        ) != config_hash(canonical_config):
            raise ValueError("calibration candidate config hash mismatch")
        if candidate.get("grid_file_hash") != run_manifest.get("lambda_grid_hash"):
            raise ValueError("calibration candidate grid hash mismatch")
        if candidate.get("source_data_hash") != run_manifest.get("input_parquet_sha256"):
            raise ValueError("calibration candidate source hash mismatch")
        if run_manifest.get("governance_profile") != GOVERNED_CALIBRATION_PROFILE:
            raise ValueError("calibration candidate governance profile mismatch")
        if bool(run_manifest.get("git_worktree_dirty")):
            raise ValueError("calibration candidate cannot originate from a dirty worktree")
        vintage = run_manifest.get("verified_vintage_catalog")
        if not isinstance(vintage, Mapping) or vintage.get("status") != (
            "VERIFIED_SIGNED_IMMUTABLE_VINTAGES"
        ):
            raise ValueError("calibration candidate vintage evidence mismatch")
    return dict(inventory)


def load_lambda_grid(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("lambda grid YAML must contain a mapping")
    return data


def iter_candidate_configs(grid: Mapping[str, object]) -> Iterable[LambdaCandidateConfig]:
    config_grid = dict(grid.get("grid", grid))
    defaults = dict(grid.get("defaults", {}))
    keys = ["lambda_smooth_month", "lambda_smooth_yoy", "lambda_shape", "neighbor_shrinkage"]
    value_lists: list[list[object]] = []
    for key in keys:
        raw = config_grid.get(key, defaults.get(key))
        if raw is None:
            raise ValueError(f"missing lambda grid key {key!r}")
        value_lists.append(list(raw if isinstance(raw, list) else [raw]))

    lookbacks = config_grid.get("history_lookback_years", defaults.get("history_lookback_years", 6))
    lookback_values = list(lookbacks if isinstance(lookbacks, list) else [lookbacks])

    baseline = LambdaCandidateConfig(
        monthly_config=MonthlyCurveConfig(
            lambda_prior=float(defaults.get("lambda_prior", 1e-6)),
            lambda_smooth_month=1.0,
            lambda_smooth_yoy=0.0,
            lambda_shape=0.0,
            neighbor_shrinkage=0.5,
            min_history_snapshots=int(defaults.get("min_history_snapshots", 24)),
            constraint_tolerance=float(defaults.get("constraint_tolerance", 0.01)),
            stationarity_tolerance=float(defaults.get("stationarity_tolerance", 1e-7)),
        ),
        history_lookback_years=int(lookback_values[0]) if lookback_values[0] is not None else None,
    )
    yield baseline

    for smooth_month in value_lists[0]:
        for smooth_yoy in value_lists[1]:
            for shape in value_lists[2]:
                for shrinkage in value_lists[3]:
                    for lookback in lookback_values:
                        yield LambdaCandidateConfig(
                            monthly_config=MonthlyCurveConfig(
                                lambda_prior=float(defaults.get("lambda_prior", 1e-6)),
                                lambda_smooth_month=float(smooth_month),
                                lambda_smooth_yoy=float(smooth_yoy),
                                lambda_shape=float(shape),
                                neighbor_shrinkage=float(shrinkage),
                                min_history_snapshots=int(defaults.get("min_history_snapshots", 24)),
                                constraint_tolerance=float(defaults.get("constraint_tolerance", 0.01)),
                                stationarity_tolerance=float(defaults.get("stationarity_tolerance", 1e-7)),
                            ),
                            history_lookback_years=int(lookback) if lookback is not None else None,
                        )


def normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "product", "load_type", "market", "price"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ValueError(f"missing required EEX history columns: {missing}")
    df = history.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["product"] = df["product"].astype(str)
    df["load_type"] = df["load_type"].astype(str).str.upper()
    df["market"] = df["market"].astype(str).str.upper()
    df["price"] = df["price"].astype(float)
    if "source" not in df.columns:
        df["source"] = ""
    sort_columns = [
        column
        for column in ("available_at", "date", "market", "load_type", "product")
        if column in df.columns
    ]
    return df.sort_values(sort_columns).reset_index(drop=True)


def select_origins(
    history: pd.DataFrame,
    *,
    settings: LambdaCalibrationSettings,
    max_origins: int | None = None,
) -> list[pd.Timestamp]:
    sub = history[
        (history["market"].eq(settings.market.upper()))
        & (history["load_type"].eq(settings.load_type.upper()))
        & (history["product"].astype(str).str.match(f"{_MONTH_RE}|{_QUARTER_RE}"))
    ]
    origin_column = "available_at" if "available_at" in sub.columns else "date"
    dates = sorted(pd.Timestamp(d) for d in sub[origin_column].dropna().unique())
    if max_origins is not None:
        dates = dates[-int(max_origins) :]
    return dates


def quote_set_for_origin(history: pd.DataFrame, origin: pd.Timestamp) -> tuple[MarketQuote, ...]:
    if "available_at" in history.columns:
        origin = _aware_utc(origin)
        available = pd.to_datetime(history["available_at"], utc=True)
        snap = history[available.eq(origin)]
    else:
        origin = pd.Timestamp(origin).tz_localize(None).normalize()
        snap = history[history["date"].eq(origin)]
    return tuple(
        MarketQuote(
            market=str(row.market).upper(),
            product=str(row.product),
            load_type=str(row.load_type).upper(),
            price=float(row.price),
            snapshot_date=pd.Timestamp(row.date),
            source=str(getattr(row, "source", "")),
            available_at=pd.Timestamp(getattr(row, "available_at", origin)),
            quote_id=str(getattr(row, "quote_id", "")),
            snapshot_id=str(getattr(row, "snapshot_id", "")),
            source_kind="EEX_HISTORICAL_VINTAGE",
            source_sha256=str(getattr(row, "source_document_sha256", "")),
        )
        for row in snap.itertuples(index=False)
        if np.isfinite(float(row.price))
    )


def select_withheld_products(
    full_quote_set: Sequence[MarketQuote],
    *,
    settings: LambdaCalibrationSettings,
    max_products: int,
) -> tuple[WithheldProduct, ...]:
    candidates: list[WithheldProduct] = []
    for quote in full_quote_set:
        if quote.market.upper() != settings.market.upper():
            continue
        if quote.load_type.upper() != settings.load_type.upper():
            continue
        tenor = product_tenor(quote.product)
        if tenor not in {"monthly", "quarterly"}:
            continue
        if quote.snapshot_date is None:
            continue
        candidates.append(
            WithheldProduct(
                market=quote.market.upper(),
                product=quote.product,
                load_type=quote.load_type.upper(),
                price=float(quote.price),
                origin_date=pd.Timestamp(
                    quote.available_at
                    if quote.available_at is not None
                    else quote.snapshot_date
                ),
                source=quote.source,
                snapshot_id=quote.snapshot_id,
            )
        )
    candidates.sort(key=lambda item: (item.tenor, item.product))
    monthly = [item for item in candidates if item.tenor == "monthly"]
    quarterly = [item for item in candidates if item.tenor == "quarterly"]
    selected: list[WithheldProduct] = []
    selected.extend(monthly[: max(0, max_products // 2)])
    selected.extend(quarterly[: max(0, max_products - len(selected))])
    if len(selected) < max_products:
        selected.extend([item for item in candidates if item not in selected][: max_products - len(selected)])
    return tuple(selected[:max_products])


@dataclass(frozen=True)
class MaskedQuoteSets:
    origin_date: pd.Timestamp
    full_quote_set: tuple[MarketQuote, ...]
    withheld_set: tuple[WithheldProduct, ...]
    visible_quote_set: tuple[MarketQuote, ...]
    own_quotes: tuple[MarketQuote, ...]
    neighbor_quotes: tuple[MarketQuote, ...]
    removed_quote_keys: tuple[str, ...]
    removed_reasons: Mapping[str, str]


def mask_quote_sets(
    full_quote_set: Sequence[MarketQuote],
    withheld: WithheldProduct,
    *,
    neighbor_markets: Sequence[str],
) -> MaskedQuoteSets:
    forbidden = revealing_quote_keys(withheld)
    neighbor_set = {m.upper() for m in neighbor_markets}
    visible: list[MarketQuote] = []
    removed: dict[str, str] = {}
    for quote in full_quote_set:
        key = quote_key(quote.market, quote.load_type, quote.product)
        same_load = quote.load_type.upper() == withheld.load_type.upper()
        same_market = quote.market.upper() == withheld.market.upper()
        is_neighbor = quote.market.upper() in neighbor_set
        if same_load and (same_market or is_neighbor) and key in forbidden:
            removed[key] = "withheld_or_revealing_product"
            continue
        visible.append(quote)
    own = tuple(
        q for q in visible
        if q.market.upper() == withheld.market.upper() and q.load_type.upper() == withheld.load_type.upper()
    )
    neighbors = tuple(q for q in visible if q.market.upper() in neighbor_set and q.load_type.upper() == withheld.load_type.upper())
    return MaskedQuoteSets(
        origin_date=withheld.origin_date,
        full_quote_set=tuple(full_quote_set),
        withheld_set=(withheld,),
        visible_quote_set=tuple(visible),
        own_quotes=own,
        neighbor_quotes=neighbors,
        removed_quote_keys=tuple(sorted(removed)),
        removed_reasons=removed,
    )


def revealing_quote_keys(withheld: WithheldProduct) -> set[str]:
    markets = {withheld.market.upper(), "DE", "FR", "AT", "IT"}
    products = revealing_products(withheld.product)
    return {
        quote_key(market, withheld.load_type, product)
        for market in markets
        for product in products
    }


def revealing_products(product: str) -> set[str]:
    product = str(product)
    tenor = product_tenor(product)
    months = product_periods(product)
    year = int(months[0].year)
    products = {product, str(year)}
    quarters = {f"{month.year}-Q{((month.month - 1) // 3) + 1}" for month in months}
    products.update(quarters)
    if tenor == "quarterly":
        products.update(str(month) for month in months)
    return products


def validate_masked_inputs(masked: MaskedQuoteSets, withheld: WithheldProduct) -> None:
    forbidden = revealing_quote_keys(withheld)
    for quote in masked.own_quotes + masked.neighbor_quotes:
        key = quote_key(quote.market, quote.load_type, quote.product)
        if key in forbidden:
            raise ValueError(f"masked quote leakage detected for {key}")
        if quote.available_at is not None and _aware_utc(quote.available_at) > _aware_utc(
            withheld.origin_date
        ):
            raise ValueError(f"future quote leaked into calibration inputs: {key}")


def history_before_origin(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    lookback_years: int | None,
) -> pd.DataFrame:
    if "available_at" in history.columns:
        origin = _aware_utc(origin)
        available = pd.to_datetime(history["available_at"], utc=True)
        hist = history[available < origin].copy()
        if lookback_years is not None:
            lower = origin - pd.DateOffset(years=int(lookback_years))
            hist = hist[pd.to_datetime(hist["available_at"], utc=True) >= lower]
    else:
        origin = pd.Timestamp(origin).tz_localize(None).normalize()
        hist = history[history["date"] < origin].copy()
        if lookback_years is not None:
            hist = hist[hist["date"] >= origin - pd.DateOffset(years=int(lookback_years))]
    return hist


def validate_history_point_in_time(history_view: pd.DataFrame, origin: pd.Timestamp) -> None:
    if history_view.empty:
        return
    if "available_at" in history_view.columns:
        origin = _aware_utc(origin)
        max_available = pd.to_datetime(history_view["available_at"], utc=True).max()
        if max_available >= origin:
            raise ValueError(
                "history feature leakage: "
                f"max available_at {max_available} >= origin {origin}"
            )
        return
    origin = pd.Timestamp(origin).tz_localize(None).normalize()
    max_date = pd.to_datetime(history_view["date"]).dt.tz_localize(None).max()
    if max_date >= origin:
        raise ValueError(f"history feature leakage: max history date {max_date} >= origin {origin}")


def validate_feature_frame_point_in_time(
    features: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    withheld: WithheldProduct,
) -> None:
    if features.empty:
        return
    if "date" in features.columns:
        dates = pd.to_datetime(features["date"]).dt.tz_localize(None)
        if bool((dates >= pd.Timestamp(origin).tz_localize(None).normalize()).any()):
            raise ValueError("pre-mask feature cache includes same-origin or future rows")
    if "product" in features.columns and bool(features["product"].astype(str).eq(withheld.product).any()):
        raise ValueError("pre-mask feature cache includes the withheld product")


def summarize_calibration(
    scoring: pd.DataFrame,
    *,
    configs: Sequence[MonthlyCurveConfig | LambdaCandidateConfig],
    settings: LambdaCalibrationSettings,
    withheld_counts: Counter[tuple[str, str, str]],
    excluded_reasons: Counter[str],
) -> dict[str, object]:
    final_status = _final_status(scoring, settings=settings)
    by_config = _config_metric_table(scoring)
    baseline_hash = config_hash(configs[0])
    best_hash = _best_config_hash(scoring, baseline_hash=baseline_hash)
    baseline_mae = _mae_for_config(scoring, baseline_hash)
    best_mae = _mae_for_config(scoring, best_hash) if best_hash else np.nan
    selection_reason = _selection_reason(
        final_status=final_status,
        baseline_mae=baseline_mae,
        best_mae=best_mae,
        baseline_hash=baseline_hash,
        best_hash=best_hash,
    )
    return {
        "final_status": final_status,
        "production_approved": False,
        "n_scoring_rows": int(len(scoring)),
        "n_origins_evaluated": int(scoring["origin_date"].nunique()) if not scoring.empty else 0,
        "n_configs_evaluated": int(scoring["config_hash"].nunique()) if not scoring.empty else 0,
        "sample_counts": _counter_to_nested_counts(withheld_counts),
        "excluded_case_counts": dict(excluded_reasons),
        "baseline_config_hash": baseline_hash,
        "selected_config_hash": best_hash,
        "baseline_mae": None if not np.isfinite(baseline_mae) else float(baseline_mae),
        "selected_mae": None if not np.isfinite(best_mae) else float(best_mae),
        "selection_reason": selection_reason,
        "train_deploy_gap": _train_deploy_gap_summary(scoring),
        "by_tenor_horizon": _tenor_horizon_metric_table(scoring),
        "by_config": by_config,
    }


def build_calibration_manifest(
    *,
    history: pd.DataFrame,
    scoring: pd.DataFrame,
    summary: Mapping[str, object],
    settings: LambdaCalibrationSettings,
    grid: Mapping[str, object],
    source_file_hash: str,
    command_line: Sequence[str],
    input_parquet_path: str,
    withheld_counts: Counter[tuple[str, str, str]],
    excluded_reasons: Counter[str],
    governance_profile: str = "UNVERIFIED_INTERNAL_DIAGNOSTIC",
    execution_environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    row_counts = {
        str(date.date() if hasattr(date, "date") else date): int(count)
        for date, count in history.groupby("date", sort=True).size().items()
    }
    execution_environment = dict(
        execution_environment or _execution_environment_receipt()
    )
    return {
        "git_commit": execution_environment["git_commit"],
        "git_worktree_dirty": execution_environment["git_worktree_dirty"],
        "command_line": list(command_line),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_parquet_path": input_parquet_path,
        "input_parquet_sha256": source_file_hash,
        "lambda_grid_hash": lambda_grid_hash(grid),
        "run_config_hash": _sha256_text(canonical_json(_settings_payload(settings))),
        "python_version": execution_environment["python_version"],
        "governance_profile": governance_profile,
        "execution_environment": execution_environment,
        "row_counts_by_snapshot": row_counts,
        "rolling_origin_provenance": dict(
            history.attrs.get("rolling_origin_provenance", {})
        ),
        "verified_vintage_catalog": dict(
            history.attrs.get("verified_vintage_catalog", {})
        ),
        "origins_evaluated": int(scoring["origin_date"].nunique()) if not scoring.empty else 0,
        "withheld_products_by_market_load_type_tenor": _counter_to_nested_counts(withheld_counts),
        "withheld_products_by_tenor_horizon": _scoring_count_table(
            scoring,
            keys=("withheld_tenor", "withheld_horizon_bucket"),
        ),
        "excluded_cases_by_reason": dict(excluded_reasons),
        "final_status": str(summary["final_status"]),
        "production_approved": False,
    }


def build_candidate_config(
    scoring: pd.DataFrame,
    *,
    summary: Mapping[str, object],
    configs: Sequence[MonthlyCurveConfig | LambdaCandidateConfig],
    settings: LambdaCalibrationSettings | None = None,
    grid_hash: str,
    source_data_hash: str,
    withheld_counts: Counter[tuple[str, str, str]],
    excluded_reasons: Counter[str],
) -> dict[str, object] | None:
    final_status = str(summary["final_status"])
    if not final_status.startswith("PASS_CALIBRATION_CANDIDATE"):
        return None
    selected_hash = str(summary.get("selected_config_hash") or config_hash(configs[0]))
    selected = next((cfg for cfg in configs if config_hash(cfg) == selected_hash), configs[0])
    active_payload = _active_config_payload(selected, settings or LambdaCalibrationSettings())
    return {
        "config_hash": config_hash(active_payload),
        "canonical_config": active_payload,
        "grid_file_hash": grid_hash,
        "source_data_hash": source_data_hash,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "selection_status": "SELECTED",
        "production_approved": False,
        "governed_scope": {
            "market": str((settings or LambdaCalibrationSettings()).market),
            "load_type": str((settings or LambdaCalibrationSettings()).load_type),
            "timezone": str((settings or LambdaCalibrationSettings()).timezone),
            "profile": GOVERNED_CALIBRATION_PROFILE,
        },
        "selection_reason": str(summary.get("selection_reason", "")),
        "sample_counts": _counter_to_nested_counts(withheld_counts),
        "excluded_config_counts": dict(excluded_reasons),
        "baseline_comparison": {
            "baseline_config_hash": summary.get("baseline_config_hash"),
            "baseline_mae": summary.get("baseline_mae"),
            "selected_mae": summary.get("selected_mae"),
            "by_tenor_horizon": summary.get("by_tenor_horizon", []),
        },
        "known_bad_known_coherent_gate_summary": {
            "available": False,
            "reason": "not part of offline lambda calibration harness",
        },
    }


def quote_key(market: str, load_type: str, product: str) -> str:
    return f"{str(market).upper()}:{str(load_type).upper()}:{str(product)}"


def product_tenor(product: str) -> str:
    text = str(product)
    if pd.Series([text]).str.match(_MONTH_RE).iloc[0]:
        return "monthly"
    if pd.Series([text]).str.match(_QUARTER_RE).iloc[0]:
        return "quarterly"
    if pd.Series([text]).str.match(_YEAR_RE).iloc[0]:
        return "calendar"
    return "unsupported"


def product_horizon_years(origin_date: pd.Timestamp, product: str) -> int:
    periods = product_periods(product)
    origin = pd.Timestamp(origin_date).tz_localize(None).normalize()
    return int(periods[0].year) - int(origin.year)


def horizon_bucket(horizon_years: int) -> str:
    if horizon_years <= 0:
        return "h+0"
    if horizon_years == 1:
        return "h+1"
    if horizon_years == 2:
        return "h+2"
    return "h+3+"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_one_case(
    *,
    masked: MaskedQuoteSets,
    withheld: WithheldProduct,
    config: MonthlyCurveConfig | LambdaCandidateConfig,
    settings: LambdaCalibrationSettings,
    history_view: pd.DataFrame,
    source_file_hash: str,
) -> dict[str, object]:
    monthly_config = _monthly_config(config)
    months = _delivery_months_for_withheld(withheld)
    horizon_years = product_horizon_years(withheld.origin_date, withheld.product)
    horizon = horizon_bucket(horizon_years)
    try:
        constraints = build_monthly_constraint_system(
            months,
            masked.own_quotes,
            timezone=settings.timezone,
            market=settings.market,
            load_type=settings.load_type,
            constraint_tolerance=monthly_config.constraint_tolerance,
        )
        shape_prior = _build_shape_prior(
            constraints=constraints,
            masked=masked,
            config=monthly_config,
            settings=settings,
            history_view=history_view,
        )
        result = solve_monthly_forward_curve_from_constraints(
            constraints,
            config=monthly_config,
            shape_prior=shape_prior,
        )
        predicted = _product_average(result.monthly_curve, withheld.product, constraints.delivery_grid.month_hours)
        signed_error = float(predicted) - float(withheld.price)
        residual_max = float(result.kkt["max_abs_constraint_residual"])
        audit = audit_monthly_curve_shape(result.monthly_curve, constraints)
        critical_count = int(audit["status"].astype(str).eq("CRITICAL").sum()) if not audit.empty else 0
        unsupported_count = int(audit["status"].astype(str).eq("UNSUPPORTED").sum()) if not audit.empty else 0
        row_status = "PASS"
        reason = ""
        if residual_max > settings.hard_constraint_tolerance:
            row_status = "FAIL_HARD_CONSTRAINT_VIOLATION"
            reason = "hard constraint residual exceeds tolerance"
        elif critical_count > 0:
            row_status = "CRITICAL_GATES"
            reason = "critical audit gates present"
        return {
            "config_hash": config_hash(config),
            "origin_date": _origin_iso(withheld.origin_date),
            "snapshot_id": withheld.snapshot_id,
            "source_file_hash": source_file_hash,
            "market": withheld.market,
            "withheld_product": withheld.product,
            "withheld_load_type": withheld.load_type,
            "withheld_tenor": withheld.tenor,
            "withheld_horizon_years": horizon_years,
            "withheld_horizon_bucket": horizon,
            "product_start": str(product_periods(withheld.product)[0]),
            "product_end": str(product_periods(withheld.product)[-1]),
            "target_price": float(withheld.price),
            "predicted_price": float(predicted),
            "abs_error": abs(signed_error),
            "signed_error": signed_error,
            "constraint_residual_max": residual_max,
            "curvature_score": _curvature_score(result.monthly_curve),
            "same_month_rank_score": _same_month_rank_score(audit),
            "historical_outlier_score": _historical_outlier_score(shape_prior),
            "neighbor_disagreement_score": _neighbor_disagreement_score(shape_prior),
            "unsupported_gate_count": unsupported_count,
            "critical_gate_count": critical_count,
            "status": row_status,
            "unsupported_reason": reason,
            "sample_size": int(len(history_view["date"].unique())) if not history_view.empty else 0,
        }
    except Exception as exc:
        return {
            "config_hash": config_hash(config),
            "origin_date": _origin_iso(withheld.origin_date),
            "snapshot_id": withheld.snapshot_id,
            "source_file_hash": source_file_hash,
            "market": withheld.market,
            "withheld_product": withheld.product,
            "withheld_load_type": withheld.load_type,
            "withheld_tenor": withheld.tenor,
            "withheld_horizon_years": horizon_years,
            "withheld_horizon_bucket": horizon,
            "product_start": str(product_periods(withheld.product)[0]),
            "product_end": str(product_periods(withheld.product)[-1]),
            "target_price": float(withheld.price),
            "predicted_price": np.nan,
            "abs_error": np.nan,
            "signed_error": np.nan,
            "constraint_residual_max": np.nan,
            "curvature_score": np.nan,
            "same_month_rank_score": np.nan,
            "historical_outlier_score": np.nan,
            "neighbor_disagreement_score": np.nan,
            "unsupported_gate_count": 1,
            "critical_gate_count": 0,
            "status": "UNSUPPORTED",
            "unsupported_reason": str(exc),
            "sample_size": int(len(history_view["date"].unique())) if not history_view.empty else 0,
        }


def _build_shape_prior(
    *,
    constraints,
    masked: MaskedQuoteSets,
    config: MonthlyCurveConfig,
    settings: LambdaCalibrationSettings,
    history_view: pd.DataFrame,
) -> MonthlyShapePrior:
    neighbor_prices: dict[str, dict[str, float]] = {}
    for quote in masked.neighbor_quotes:
        neighbor_prices.setdefault(quote.market.upper(), {})[quote.product] = float(quote.price)
    panel = build_neighbor_panel_shape_prior(
        constraints,
        neighbor_prices,
        neighbor_markets=settings.neighbor_markets,
        neighbor_shrinkage=config.neighbor_shrinkage,
        run_timestamp=masked.origin_date,
    )
    history = build_history_shape_prior(
        constraints,
        history_view,
        market=settings.market,
        load_type=settings.load_type,
        run_timestamp=masked.origin_date - pd.Timedelta(days=1),
        min_snapshots=config.min_history_snapshots,
        lookback_years=settings.history_lookback_years,
    )
    structural = build_structural_monthly_shape_prior_from_history(
        constraints,
        history_view,
        market=settings.market,
        load_type=settings.load_type,
        run_timestamp=masked.origin_date - pd.Timedelta(days=1),
        min_snapshots=settings.min_structural_snapshots,
        lookback_years=settings.history_lookback_years,
        fallback_to_template=settings.allow_template_structural_fallback,
        fallback_amplitude_eur_mwh=settings.structural_amplitude_eur_mwh,
    )
    return build_fused_shape_prior(
        constraints,
        panel_prior=panel,
        history_prior=history,
        structural_prior=structural,
        weights={
            "panel": settings.panel_weight,
            "history": settings.history_weight,
            "structural": settings.structural_weight,
        },
    )


def _delivery_months_for_withheld(withheld: WithheldProduct) -> pd.PeriodIndex:
    periods = product_periods(withheld.product)
    year = int(periods[0].year)
    return pd.period_range(f"{year}-01", f"{year}-12", freq="M")


def _product_average(curve: pd.Series, product: str, hours: pd.Series) -> float:
    months = product_periods(product)
    values = curve.reindex(months).astype(float)
    weights = hours.reindex(months).astype(float)
    if values.isna().any() or weights.isna().any():
        raise ValueError(f"curve does not cover withheld product {product}")
    return float((values.to_numpy() * weights.to_numpy()).sum() / weights.sum())


def _curvature_score(curve: pd.Series) -> float:
    arr = curve.to_numpy(dtype=float)
    if len(arr) < 3:
        return 0.0
    return float(np.mean(np.diff(arr, n=2) ** 2))


def _same_month_rank_score(audit: pd.DataFrame) -> float:
    if audit.empty or "metric_value" not in audit.columns:
        return 0.0
    sub = audit[audit["gate_id"].astype(str).eq("same_month_rank_consistency")]
    if sub.empty:
        return 0.0
    return float(sub["metric_value"].astype(float).abs().max())


def _historical_outlier_score(prior: MonthlyShapePrior) -> float:
    diag = prior.diagnostics
    if diag.empty or "mad_deviation" not in diag.columns:
        return 0.0 if prior.status != "UNSUPPORTED" else np.nan
    mad = pd.to_numeric(diag["mad_deviation"], errors="coerce").replace(0.0, np.nan)
    med = pd.to_numeric(diag.get("median_deviation", pd.Series(dtype=float)), errors="coerce")
    if mad.dropna().empty:
        return 0.0
    return float((med.abs() / mad).replace([np.inf, -np.inf], np.nan).max(skipna=True))


def _neighbor_disagreement_score(prior: MonthlyShapePrior) -> float:
    contrib = prior.contributions
    if contrib.empty or contrib.shape[1] < 2:
        return 0.0
    return float(contrib.std(axis=1, skipna=True).mean(skipna=True))


def _final_status(scoring: pd.DataFrame, *, settings: LambdaCalibrationSettings) -> str:
    if scoring.empty:
        return "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS"
    if bool(scoring["status"].astype(str).eq("FAIL_HARD_CONSTRAINT_VIOLATION").any()):
        return "FAIL_HARD_CONSTRAINT_VIOLATION"
    if bool((pd.to_numeric(scoring["constraint_residual_max"], errors="coerce") > settings.hard_constraint_tolerance).any()):
        return "FAIL_HARD_CONSTRAINT_VIOLATION"
    if bool((pd.to_numeric(scoring["critical_gate_count"], errors="coerce").fillna(0) > 0).any()):
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    origins = int(scoring["origin_date"].nunique())
    monthly = int(scoring[scoring["withheld_tenor"].eq("monthly")]["withheld_product"].nunique())
    quarterly = int(scoring[scoring["withheld_tenor"].eq("quarterly")]["withheld_product"].nunique())
    if origins < settings.min_valid_origins:
        return "UNSUPPORTED_INSUFFICIENT_HISTORY"
    if monthly < settings.min_withheld_monthly or quarterly < settings.min_withheld_quarterly:
        return "UNSUPPORTED_TOO_FEW_WITHHELD_PRODUCTS"
    baseline = str(scoring["config_hash"].iloc[0])
    best = _best_config_hash(scoring, baseline_hash=baseline)
    if best is None:
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    baseline_mae = _mae_for_config(scoring, baseline)
    best_mae = _mae_for_config(scoring, best)
    if not np.isfinite(baseline_mae) or not np.isfinite(best_mae):
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    abs_improvement = baseline_mae - best_mae
    rel_improvement = abs_improvement / max(abs(baseline_mae), 1e-9)
    if best == baseline or (
        abs_improvement < settings.identifiability_min_abs_error_improvement
        and rel_improvement < settings.identifiability_min_rel_error_improvement
    ):
        return "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA"
    return "PASS_CALIBRATION_CANDIDATE_NOT_PRODUCTION_APPROVED"


def _best_config_hash(scoring: pd.DataFrame, *, baseline_hash: str) -> str | None:
    grouped = _complete_config_mae(scoring)
    if grouped.empty:
        return None
    return str(grouped.sort_values(kind="mergesort").index[0])


def _mae_for_config(scoring: pd.DataFrame, cfg_hash: str | None) -> float:
    if not cfg_hash or scoring.empty:
        return float("nan")
    complete = _complete_config_mae(scoring)
    return float(complete.loc[str(cfg_hash)]) if str(cfg_hash) in complete.index else float("nan")


def _complete_config_mae(scoring: pd.DataFrame) -> pd.Series:
    """Score only configurations covering the exact same finite PASS case set."""

    required = {
        "config_hash",
        "origin_date",
        "snapshot_id",
        "market",
        "withheld_load_type",
        "withheld_product",
        "status",
        "abs_error",
    }
    if scoring.empty or not required <= set(scoring.columns):
        return pd.Series(dtype=float)
    case_columns = [
        "origin_date",
        "snapshot_id",
        "market",
        "withheld_load_type",
        "withheld_product",
    ]
    expected_cases = {
        tuple(str(row[column]) for column in case_columns)
        for row in scoring.to_dict(orient="records")
    }
    scores: dict[str, float] = {}
    for cfg_hash, group in scoring.groupby("config_hash", sort=True):
        cases = [
            tuple(str(row[column]) for column in case_columns)
            for row in group.to_dict(orient="records")
        ]
        errors = pd.to_numeric(group["abs_error"], errors="coerce")
        if (
            len(cases) != len(expected_cases)
            or len(set(cases)) != len(cases)
            or set(cases) != expected_cases
            or not group["status"].astype(str).eq("PASS").all()
            or not np.isfinite(errors.to_numpy(dtype=float)).all()
        ):
            continue
        scores[str(cfg_hash)] = float(errors.mean())
    return pd.Series(scores, dtype=float)


def _config_metric_table(scoring: pd.DataFrame) -> list[dict[str, object]]:
    if scoring.empty:
        return []
    rows: list[dict[str, object]] = []
    for cfg_hash, group in scoring.groupby("config_hash", sort=True):
        complete = str(cfg_hash) in _complete_config_mae(scoring).index
        rows.append(
            {
                "config_hash": str(cfg_hash),
                "mean_abs_error": float(pd.to_numeric(group["abs_error"], errors="coerce").mean()),
                "median_abs_error": float(pd.to_numeric(group["abs_error"], errors="coerce").median()),
                "max_constraint_residual": float(pd.to_numeric(group["constraint_residual_max"], errors="coerce").max()),
                "critical_gate_count": int(pd.to_numeric(group["critical_gate_count"], errors="coerce").fillna(0).sum()),
                "n_rows": int(len(group)),
                "complete_case_eligible": bool(complete),
            }
        )
    return rows


def _tenor_horizon_metric_table(scoring: pd.DataFrame) -> list[dict[str, object]]:
    required = {"withheld_tenor", "withheld_horizon_bucket", "abs_error", "config_hash"}
    if scoring.empty or not required <= set(scoring.columns):
        return []
    rows: list[dict[str, object]] = []
    for (tenor, horizon), group in scoring.groupby(["withheld_tenor", "withheld_horizon_bucket"], sort=True):
        valid = pd.to_numeric(group["abs_error"], errors="coerce").dropna()
        rows.append(
            {
                "withheld_tenor": str(tenor),
                "withheld_horizon_bucket": str(horizon),
                "n_rows": int(len(group)),
                "n_origins": int(group["origin_date"].nunique()) if "origin_date" in group.columns else 0,
                "n_products": int(group["withheld_product"].nunique()) if "withheld_product" in group.columns else 0,
                "mean_abs_error": None if valid.empty else float(valid.mean()),
                "median_abs_error": None if valid.empty else float(valid.median()),
                "min_abs_error": None if valid.empty else float(valid.min()),
                "max_abs_error": None if valid.empty else float(valid.max()),
                "n_configs": int(group["config_hash"].nunique()),
            }
        )
    return rows


def _train_deploy_gap_summary(scoring: pd.DataFrame) -> dict[str, object]:
    if scoring.empty or "withheld_horizon_bucket" not in scoring.columns:
        return {
            "status": "UNSUPPORTED_NO_SCORING",
            "message": "No withheld-product scoring is available.",
        }
    counts = _scoring_count_table(scoring, keys=("withheld_tenor", "withheld_horizon_bucket"))
    far_rows = scoring[scoring["withheld_horizon_bucket"].astype(str).isin(["h+2", "h+3+"])]
    far_monthly_rows = far_rows[far_rows["withheld_tenor"].astype(str).eq("monthly")]
    if far_monthly_rows.empty:
        return {
            "status": "UNSUPPORTED_NO_FAR_HORIZON_MONTHLY_TRUTH",
            "message": (
                "Withheld monthly/quarterly quote tests validate near-tenor reconstruction only; "
                "the sparse h+2/h+3 deployment zone has no direct withheld monthly truth in this run."
            ),
            "counts": counts,
            "far_horizon_non_monthly_rows": int(len(far_rows)),
        }
    return {
        "status": "PARTIAL_FAR_HORIZON_EVIDENCE",
        "message": "Some far-horizon withheld monthly products were scored; inspect by_tenor_horizon before promotion.",
        "counts": counts,
        "far_horizon_monthly_rows": int(len(far_monthly_rows)),
    }


def _scoring_count_table(scoring: pd.DataFrame, *, keys: tuple[str, ...]) -> dict[str, int]:
    if scoring.empty or not set(keys) <= set(scoring.columns):
        return {}
    return {
        "|".join(str(part) for part in idx if str(part) != ""): int(count)
        for idx, count in scoring.groupby(list(keys), sort=True).size().items()
    }


def _selection_reason(
    *,
    final_status: str,
    baseline_mae: float,
    best_mae: float,
    baseline_hash: str,
    best_hash: str | None,
) -> str:
    if final_status.startswith("PASS_CALIBRATION_CANDIDATE"):
        return (
            "candidate minimizes withheld-product MAE versus baseline while "
            "remaining research-only and production_approved=false"
        )
    if final_status == "UNSUPPORTED_NO_IDENTIFIABLE_LAMBDA":
        return (
            f"lambda grid did not clearly beat baseline {baseline_hash}; "
            f"baseline_mae={baseline_mae}, best_hash={best_hash}, best_mae={best_mae}"
        )
    return f"{final_status}: no production-approved lambda selected"


def _settings_from_grid(
    grid: Mapping[str, object],
    *,
    settings: LambdaCalibrationSettings | None,
    smoke: bool,
) -> LambdaCalibrationSettings:
    base = settings or LambdaCalibrationSettings()
    raw = dict(grid.get("calibration", {}))
    if smoke:
        raw.update(dict(grid.get("smoke", {})))
    if not raw:
        return base
    payload = _settings_payload(base)
    payload.update(raw)
    if isinstance(payload.get("neighbor_markets"), str):
        payload["neighbor_markets"] = tuple(m.strip().upper() for m in str(payload["neighbor_markets"]).split(",") if m.strip())
    elif "neighbor_markets" in payload:
        payload["neighbor_markets"] = tuple(str(m).upper() for m in payload["neighbor_markets"])
    return LambdaCalibrationSettings(**payload)


def _validate_governed_calibration_settings(
    settings: LambdaCalibrationSettings,
) -> None:
    violations: list[str] = []
    for field_name, value in settings.__dict__.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            continue
        if not np.isfinite(float(value)):
            violations.append(f"{field_name} is non-finite")
    for field_name, floor in _GOVERNED_MINIMUM_SETTINGS.items():
        value = getattr(settings, field_name)
        if not isinstance(value, (int, float, np.number)) or isinstance(value, bool):
            violations.append(f"{field_name} is not numeric")
        elif np.isfinite(float(value)) and value < floor:
            violations.append(f"{field_name}={value} < {floor}")
    for field_name, ceiling in _GOVERNED_MAXIMUM_SETTINGS.items():
        value = getattr(settings, field_name)
        if not isinstance(value, (int, float, np.number)) or isinstance(value, bool):
            violations.append(f"{field_name} is not numeric")
        elif np.isfinite(float(value)) and value > ceiling:
            violations.append(f"{field_name}={value} > {ceiling}")
    for field_name, expected in _GOVERNED_EXACT_SETTINGS.items():
        value = getattr(settings, field_name)
        if value != expected:
            violations.append(f"{field_name}={value!r} != {expected!r}")
    if violations:
        raise EexHistoricalVintageError(
            "governed calibration settings weaken tier2_governed_minimums.v1: "
            + "; ".join(violations)
        )


def _active_config_payload(
    config: MonthlyCurveConfig | LambdaCandidateConfig | Mapping[str, object],
    settings: LambdaCalibrationSettings,
) -> dict[str, object]:
    payload = _config_payload(config)
    payload.update(
        {
            "markets": sorted(str(market).upper() for market in settings.neighbor_markets),
            "min_structural_snapshots": int(settings.min_structural_snapshots),
            "allow_template_structural_fallback": bool(settings.allow_template_structural_fallback),
            "structural_amplitude_eur_mwh": float(settings.structural_amplitude_eur_mwh),
            "panel_weight": float(settings.panel_weight),
            "history_weight": float(settings.history_weight),
            "structural_weight": float(settings.structural_weight),
        }
    )
    return payload


def _monthly_config(config: MonthlyCurveConfig | LambdaCandidateConfig) -> MonthlyCurveConfig:
    if isinstance(config, LambdaCandidateConfig):
        return config.monthly_config
    return config


def _candidate_history_lookback(
    config: MonthlyCurveConfig | LambdaCandidateConfig,
    settings: LambdaCalibrationSettings,
) -> int | None:
    if isinstance(config, LambdaCandidateConfig):
        return config.history_lookback_years
    return settings.history_lookback_years


def _settings_payload(settings: LambdaCalibrationSettings) -> dict[str, object]:
    payload = dict(settings.__dict__)
    payload["neighbor_markets"] = tuple(settings.neighbor_markets)
    return payload


def _aware_utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("rolling-origin timestamp is invalid")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _origin_iso(value: object) -> str:
    return _aware_utc(value).isoformat()


def _counter_to_nested_counts(counter: Counter[tuple[str, str, str]]) -> dict[str, int]:
    return {"|".join(map(str, key)): int(value) for key, value in sorted(counter.items())}


def _validate_artifacts(
    scoring: pd.DataFrame,
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    candidate: Mapping[str, object] | None,
) -> None:
    final_status = str(summary.get("final_status"))
    if final_status not in FINAL_STATUSES:
        raise ValueError(f"unknown final_status {final_status!r}")
    if bool(summary.get("production_approved")) or bool(manifest.get("production_approved")):
        raise ValueError("lambda calibration artifacts must never be production-approved")
    missing = [col for col in SCORING_COLUMNS if col not in scoring.columns]
    if missing:
        raise ValueError(f"scoring artifact missing required columns: {missing}")
    if not final_status.startswith("PASS_CALIBRATION_CANDIDATE") and candidate is not None:
        raise ValueError("non-candidate status cannot produce a candidate config")
    if candidate is not None and bool(candidate.get("production_approved")):
        raise ValueError("candidate config must keep production_approved=false")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _execution_environment_receipt() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "pfc_shaping"
    source_paths = sorted(
        (
            path
            for path in package_root.rglob("*.py")
            if "ct" not in path.relative_to(package_root).parts
        ),
        key=lambda path: path.as_posix(),
    )
    source_paths.append(repo_root / "scripts/run_monthly_curve_lambda_calibration.py")
    source_files = {
        path.relative_to(repo_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    dependency_versions = sorted(
        {
            f"{distribution.metadata.get('Name', 'UNKNOWN')}=={distribution.version}"
            for distribution in importlib_metadata.distributions()
        }
    )
    dirty = _git_worktree_dirty()
    return {
        "git_commit": _git_commit(),
        "git_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "source_files": source_files,
        "source_bundle_sha256": _sha256_text(canonical_json(source_files)),
        "dependency_versions_sha256": _sha256_text(
            canonical_json(dependency_versions)
        ),
        "dependency_count": len(dependency_versions),
    }


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _git_worktree_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                text=True,
            ).strip()
        )
    except Exception:
        return True
