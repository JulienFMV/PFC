from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.pipeline.atomic_promotion as atomic_promotion
import pfc_shaping.pipeline.governed_release as governed_release
import scripts.check_monthly_curve_promotion_from_manifests as capstone
from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.calibration.monthly_forward_curve import (
    MarketQuote,
    build_monthly_constraint_system,
)
from pfc_shaping.data.acquisition_signing import sign_acquisition_contract
from pfc_shaping.data.forward_proxy import (
    load_forward_snapshot,
    validate_forward_snapshot,
)
from pfc_shaping.data.lt_input_sources import (
    dataframe_sha256,
    validate_governed_forward_history_frame,
)
from pfc_shaping.pipeline.atomic_promotion import (
    PromotionError,
    finalize_assembled_candidate_staging,
    verify_candidate_bundle,
)
from pfc_shaping.pipeline.candidate_evidence import (
    CandidateEvidenceError,
    _verified_eex_acquisition,
    capture_pre_run_governance_evidence,
    seal_assembled_candidate_evidence,
    verify_assembled_candidate_evidence,
)
from pfc_shaping.pipeline.candidate_evidence_assembler import (
    ASSEMBLY_MANIFEST,
    HOURLY_EXPORT,
    CandidateEvidenceAssemblyError,
    _canonical_frame_records,
    _delivery_quarter_hour_grid,
    _hash_frame,
    _sha256_json,
    assemble_candidate_derived_evidence,
    verify_candidate_derived_evidence,
)
from pfc_shaping.pipeline.candidate_product_evidence import (
    CONFLICT_INVENTORY,
    STAGED_POLICY,
    CandidateProductEvidenceError,
    assemble_candidate_product_evidence,
    verify_candidate_product_evidence,
)
from pfc_shaping.pipeline.governed_release import (
    GovernedReleaseError,
    audit_release_request,
    register_assembled_release_request,
)
from pfc_shaping.pipeline.model_governance_contract import (
    sign_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    active_monthly_curve_config_payload,
    monthly_solver_settings,
)
from pfc_shaping.pipeline.quality_gate import validate_input_frame
from pfc_shaping.pipeline.quote_conflict_policy_contract import (
    sign_quote_conflict_policy,
)
from pfc_shaping.pipeline.release_request_contract import ensure_workflow_domain
from scripts.assemble_lt_candidate_product_evidence import main as product_evidence_main
from scripts.finalize_lt_candidate import main as finalize_candidate_main

_SOURCE_SYSTEM_BY_ROLE = {
    "epex_ch": "EPEX_SPOT",
    "epex_de": "EPEX_SPOT",
    "entso": "ENTSOE_TRANSPARENCY_PLATFORM",
    "hydro": "FMV_HYDRO_CURATED",
    "eex_forwards_history": "EEX_MARKET_DATA",
}


def _provision_registration_roots(
    tmp_path: Path,
    *,
    workflow_name: str,
    evidence_name: str,
) -> tuple[Path, Path]:
    workflow = tmp_path / workflow_name
    evidence = tmp_path / evidence_name
    workflow.mkdir()
    evidence.mkdir()
    ensure_workflow_domain(workflow, create=True)
    return workflow, evidence


@pytest.fixture(autouse=True)
def _trusted_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PFC_PROMOTION_RELEASE_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174003",
    )
    monkeypatch.setenv(
        "PFC_RELEASE_WORKFLOW_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174004",
    )
    for prefix, private_env, public_env in (
        (
            "model",
            "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
            "PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "data",
            "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
            "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "quote-policy",
            "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
            "PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH",
        ),
        (
            "release-request",
            "PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
            "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH",
        ),
    ):
        key = Ed25519PrivateKey.generate()
        private = tmp_path / f"{prefix}-private.pem"
        public = tmp_path / f"{prefix}-public.pem"
        private.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        public.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        monkeypatch.setenv(private_env, str(private))
        monkeypatch.setenv(public_env, str(public))


def _write_historical_thresholds(path: Path) -> None:
    columns = [
        "gate_id",
        "metric",
        "market",
        "delivery_bucket",
        "lookback_start",
        "lookback_end",
        "n_snapshots",
        "min_required_n",
        "p50",
        "p90",
        "p975",
        "max_observed",
        "regime_filter",
        "status",
    ]
    rows = [",".join(columns)]
    for gate, metric in (
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        (
            "residual_vs_implied_comparable_block",
            "comparable_block_shape_delta_abs_eur_mwh",
        ),
    ):
        for bucket in ["all", *(f"month_{month:02d}" for month in range(1, 13))]:
            rows.append(f"{gate},{metric},CH,{bucket},,,0,24,,,,,fixture,UNSUPPORTED")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _governed_input_generation(
    root: Path,
    *,
    reference: pd.Timestamp,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, list[dict[str, object]]]:
    generation = root / "snapshots" / "input-001"
    generation.mkdir(parents=True)
    common_index = pd.date_range(
        reference.floor("15min") - pd.Timedelta(days=7),
        reference.floor("15min"),
        freq="15min",
    )
    frames = {
        "epex_ch": pd.DataFrame({"price_eur_mwh": 80.0}, index=common_index),
        "epex_de": pd.DataFrame({"price_eur_mwh": 75.0}, index=common_index),
        "entso": pd.DataFrame(
            {"load_mw": 7000.0, "solar_mw": 1000.0, "wind_mw": 500.0},
            index=common_index,
        ),
        "hydro": pd.DataFrame(
            {"fill_deviation": 0.0},
            index=pd.date_range(
                reference.floor("D") - pd.Timedelta(days=30),
                reference.floor("D"),
                freq="D",
            ),
        ),
    }
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-13"]),
            "product": ["2027"],
            "load_type": ["BASE"],
            "product_type": ["CAL"],
            "price": [82.0],
            "market": ["CH"],
            "source": ["EEX"],
        }
    )
    files: dict[str, object] = {}
    receipts: dict[str, object] = {}
    reports: dict[str, object] = {}
    for role, frame in frames.items():
        logical = f"inputs/{role}.parquet"
        path = generation / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        payload = path.read_bytes()
        files[role] = {
            "path": logical,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "acquisition_id": "input-001",
            "available_at_utc": "2026-07-13T23:00:00+00:00",
            "calibration_eligible": True,
            "expected_cadence_seconds": 86400 if role == "hydro" else 900,
            "source_system": _SOURCE_SYSTEM_BY_ROLE[role],
        }
        receipts[role] = {
            "role": role,
            "logical_path": logical,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(frame),
            "frame_sha256": dataframe_sha256(frame),
        }
        columns = {
            "epex_ch": ["price_eur_mwh"],
            "epex_de": ["price_eur_mwh"],
            "entso": ["load_mw", "solar_mw", "wind_mw"],
            "hydro": ["fill_deviation"],
        }[role]
        report = validate_input_frame(
            frame,
            name=role,
            required_columns=columns,
            min_rows=1,
            max_age_days=10.0 if role == "hydro" else 3.0 if role == "entso" else 2.0,
            reference_timestamp=reference,
            fail_on_stale=True,
            min_finite_fraction=0.95,
            recent_window_days=7.0,
            min_recent_finite_fraction=0.95,
            max_finite_gap_days=(
                14.0 if role == "hydro" else 2.0 / 24.0 if role == "entso" else 1.0 / 24.0
            ),
        )
        reports[role] = {
            "dataset": report.dataset,
            "checks_passed": report.checks_passed,
            "warnings": report.warnings,
            "errors": report.errors,
            "metrics": report.metrics,
        }
    history_path = generation / "inputs" / "eex_forwards_history.parquet"
    history.to_parquet(history_path, index=False)
    history_payload = history_path.read_bytes()
    history_hash = hashlib.sha256(history_payload).hexdigest()
    files["eex_forwards_history"] = {
        "path": "inputs/eex_forwards_history.parquet",
        "size_bytes": len(history_payload),
        "sha256": history_hash,
        "acquisition_id": "input-001",
        "available_at_utc": "2026-07-13T23:00:00+00:00",
        "calibration_eligible": True,
        "expected_cadence_seconds": None,
        "source_system": _SOURCE_SYSTEM_BY_ROLE["eex_forwards_history"],
    }
    receipts["eex_forwards_history"] = {
        "role": "eex_forwards_history",
        "logical_path": "inputs/eex_forwards_history.parquet",
        "size_bytes": len(history_payload),
        "sha256": history_hash,
        "rows": len(history),
        "frame_sha256": dataframe_sha256(history),
    }
    validated = validate_governed_forward_history_frame(
        history,
        reference_timestamp=reference,
    )
    return files, receipts, reports, history_hash, list(
        validated.attrs["eex_source_summary"]
    )


def _fixture(
    tmp_path: Path,
    *,
    production_hash: str | None = None,
    staging_path: Path | None = None,
    governed_data_root: Path | None = None,
    extra_config: dict[str, object] | None = None,
) -> Path:
    staging = staging_path or (tmp_path / "candidate")
    staging.mkdir(parents=True)
    authority = tmp_path / "authority"
    authority.mkdir()
    config: dict[str, object] = {
        "forwards": {
            "monthly_curve_solver": {
                "enabled": True,
                "target_markets": ["CH"],
            }
        },
        "quality": {
            "benchmark_policy": "advisory",
            "fail_on_benchmark": False,
        },
    }
    config.update(extra_config or {})
    canonical = active_monthly_curve_config_payload(monthly_solver_settings(config))
    active_hash = config_hash(canonical)
    thresholds_source = authority / "historical_thresholds.csv"
    selected_source = authority / "selected.json"
    _write_historical_thresholds(thresholds_source)
    selected_source.write_text(
        json.dumps(
            {
                "schema_version": "monthly_curve_selected_lambda_decision.v1",
                "decision_id": "decision-test",
                "calibration_campaign_id": "campaign-test",
                "selection_reason": "governed fixture",
                "selection_status": "PRODUCTION_APPROVED",
                "production_approved": True,
                "canonical_config": canonical,
                "config_hash": active_hash,
            }
        ),
        encoding="utf-8",
    )
    receipts: dict[str, Path] = {}
    for role, artifact in (
        ("historical_thresholds", thresholds_source),
        ("selected_lambda_decision", selected_source),
    ):
        receipt = authority / f"{role}.receipt.json"
        receipt.write_text(
            json.dumps(
                sign_model_governance_artifact_receipt(
                    artifact,
                    artifact_role=role,
                    authority_id="FMV_MODEL_GOVERNANCE_TEST",
                    issued_at_utc="2026-07-12T00:00:00Z",
                    data_cutoff_utc="2026-07-11T00:00:00Z",
                    private_key_path=os.environ[
                        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"
                    ],
                )
            ),
            encoding="utf-8",
        )
        receipts[role] = receipt
    eex_source = authority / "Price_Report_EEX.xlsx"
    workbook_rows = [
        [
            None,
            "M01_2027_BASE",
            "M02_2027_BASE",
            "M03_2027_BASE",
            "M01_2027_PEAK",
            "M02_2027_PEAK",
            "M03_2027_PEAK",
        ],
        [None, "ISIN-1", "ISIN-2", "ISIN-3", "ISIN-4", "ISIN-5", "ISIN-6"],
        ["Date", None, None, None, None, None, None],
        ["13.07.2026", 83.5, 82.0, 81.0, 83.5, 82.0, 81.0],
    ]
    with pd.ExcelWriter(eex_source, engine="openpyxl") as writer:
        pd.DataFrame(workbook_rows).to_excel(
            writer,
            sheet_name="CH",
            index=False,
            header=False,
        )
    eex_contract = authority / "eex-acquisition.json"
    eex_contract.write_text(
        json.dumps(
            sign_acquisition_contract(
                {
                    "schema_version": "lt_input_snapshot.v1",
                    "layout": "external_v2",
                    "generation_id": "eex-001",
                    "acquisition_id": "eex-001",
                    "calibration_eligible": True,
                    "source_class": "GOVERNED_ACQUISITION",
                    "files": {
                        "eex_forward_source": {
                            "path": eex_source.name,
                            "size_bytes": eex_source.stat().st_size,
                            "sha256": hashlib.sha256(eex_source.read_bytes()).hexdigest(),
                            "acquisition_id": "eex-001",
                            "available_at_utc": "2026-07-13T18:00:00+00:00",
                            "calibration_eligible": True,
                            "source_system": "EEX_MARKET_DATA",
                        }
                    },
                },
                private_key_path=os.environ[
                    "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    runtime_config = authority / "runtime-config.yaml"
    runtime_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    capture_pre_run_governance_evidence(
        staging,
        run_id="run-001",
        valuation_timestamp="2026-07-13T23:59:59+00:00",
        build_timestamp="2026-07-14T00:00:00+00:00",
        historical_thresholds_path=thresholds_source,
        historical_thresholds_receipt_path=receipts["historical_thresholds"],
        selected_lambda_decision_path=selected_source,
        selected_lambda_decision_receipt_path=receipts["selected_lambda_decision"],
        eex_forward_source_path=eex_source,
        eex_acquisition_contract_path=eex_contract,
        runtime_config_path=runtime_config,
        expected_runtime_config_sha256=hashlib.sha256(
            runtime_config.read_bytes()
        ).hexdigest(),
        source_revision="1" * 40,
        input_snapshot_sha256="2" * 64,
        input_pointer_sha256="3" * 64,
        input_generation_id="generation-001",
        peak_source_policy="same_first",
        use_seasonal_hourly_shape=True,
    )

    (staging / "markets" / "CH").mkdir(parents=True)
    (staging / "evidence" / "config").mkdir(parents=True)
    (staging / "evidence" / "data").mkdir(parents=True)
    (staging / "evidence" / "eex").mkdir(parents=True)

    config_path = staging / "evidence" / "config" / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    pre_run_path = staging / "manifests" / "pre_run_governance_manifest.json"

    snapshot = staging / "evidence" / "data" / "lt_input_snapshot.json"
    if governed_data_root is None:
        input_files = {
            role: {
                "path": f"curated/{role}.parquet",
                "size_bytes": len(role.encode("utf-8")),
                "sha256": hashlib.sha256(role.encode("utf-8")).hexdigest(),
                "acquisition_id": "input-001",
                "available_at_utc": "2026-07-12T18:00:00+00:00",
                "calibration_eligible": True,
                "source_system": _SOURCE_SYSTEM_BY_ROLE[role],
            }
            for role in ("epex_ch", "epex_de", "entso", "hydro", "eex_forwards_history")
        }
        input_receipts = {
            role: {
                "role": role,
                "logical_path": entry["path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
                "rows": 1,
                "frame_sha256": hashlib.sha256(
                    f"frame:{role}".encode("utf-8")
                ).hexdigest(),
            }
            for role, entry in input_files.items()
        }
        freshness_reports: dict[str, object] = {}
        history_hash = ""
        history_summary: list[dict[str, object]] = []
    else:
        (
            input_files,
            input_receipts,
            freshness_reports,
            history_hash,
            history_summary,
        ) = _governed_input_generation(
            governed_data_root,
            reference=pd.Timestamp("2026-07-13T23:59:59Z"),
        )
    snapshot.write_text(
        json.dumps(
            sign_acquisition_contract(
                {
                    "schema_version": "lt_input_snapshot.v1",
                    "layout": "external_v2",
                    "generation_id": "input-001",
                    "acquisition_id": "input-001",
                    "available_at_utc": (
                        "2026-07-13T23:00:00+00:00"
                        if governed_data_root is not None
                        else "2026-07-12T18:00:00+00:00"
                    ),
                    "calibration_eligible": True,
                    "source_class": "GOVERNED_ACQUISITION",
                    "files": input_files,
                },
                private_key_path=os.environ[
                    "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    if governed_data_root is not None:
        live_contract = (
            governed_data_root / "snapshots" / "input-001" / "lt_input_snapshot.json"
        )
        live_contract.write_bytes(snapshot.read_bytes())
    pointer = staging / "evidence" / "data" / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": "lt_data_pointer.v1",
                "generation_id": "input-001",
                "contract_path": "snapshots/input-001/lt_input_snapshot.json",
                "contract_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    source = staging / "evidence" / "eex" / "source.xlsx"
    source.write_bytes(eex_source.read_bytes())
    spot = pd.DataFrame(
        {"price_eur_mwh": [80.0]},
        index=pd.date_range("2026-07-01", periods=1, freq="15min", tz="UTC"),
    )
    with patch(
        "pfc_shaping.data.forward_proxy._observation_timestamp_utc",
        return_value=pd.Timestamp("2026-07-13T18:00:00Z"),
    ):
        forward_snapshot = load_forward_snapshot(
            spot,
            eex_report_path=str(eex_source),
            config={},
            market="CH",
            allow_spot_proxy=False,
        )
    forward_eligibility = validate_forward_snapshot(
        forward_snapshot,
        reference_timestamp="2026-07-13T23:59:59+00:00",
    )
    quotes = staging / "evidence" / "eex" / "forward_quote_snapshot.parquet"
    forward_manifest = forward_snapshot.to_manifest()
    forward_manifest["source_path"] = "../evidence/eex/source.xlsx"
    forward_manifest["source_description"] = "archived EEX source in test candidate bundle"
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-13"),
                "product": str(quote["product"]).removesuffix("-Peak").removesuffix("-Offpeak"),
                "load_type": str(quote["load_type"]),
                "product_type": "Month",
                "price": float(quote["price_eur_mwh"]),
                "market": "CH",
                "source": "candidate_forward_snapshot",
                "quote_id": str(quote["quote_id"]),
            }
            for quote in forward_manifest["quotes"]
        ]
    ).to_parquet(quotes, index=False)
    delivery_months = pd.period_range("2027-01", "2027-03", freq="M")
    pfc_index = pd.date_range(
        pd.Timestamp("2027-01-01", tz="Europe/Zurich").tz_convert("UTC"),
        pd.Timestamp("2027-04-01", tz="Europe/Zurich").tz_convert("UTC"),
        freq="15min",
        inclusive="left",
    )
    local_month = pfc_index.tz_convert("Europe/Zurich").month
    month_prices = {1: 83.5, 2: 82.0, 3: 81.0}
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pd.DataFrame(
        {"price_shape": [month_prices[int(month)] for month in local_month]},
        index=pfc_index,
    ).to_parquet(pfc_path)
    solver_quotes = [
        MarketQuote(
            market="CH",
            product=str(quote["product"]),
            load_type="BASE",
            price=float(quote["price_eur_mwh"]),
            snapshot_date=pd.Timestamp(forward_manifest["snapshot_date"]),
            source=str(forward_manifest["source_kind"]),
            available_at=pd.Timestamp(forward_manifest["available_at"]),
            quote_id=str(quote["quote_id"]),
            snapshot_id=str(forward_manifest["snapshot_id"]),
            observation_id=str(forward_manifest["observation_id"]),
            source_kind=str(forward_manifest["source_kind"]),
            source_sha256=str(forward_manifest["source_sha256"]),
        )
        for quote in forward_manifest["quotes"]
        if str(quote["load_type"]).upper() == "BASE"
    ]
    constraints = build_monthly_constraint_system(
        delivery_months,
        solver_quotes,
        market="CH",
        load_type="BASE",
    )
    provenance_columns = [
        "constraint_name",
        "product",
        "parent_product",
        "source_quote_ids",
        "lineage_formula",
        "lineage_sha256",
        "lineage_payload",
        "target",
        "hours",
        "n_months",
        "is_residual",
        "load_type",
    ]
    constraint_provenance = _canonical_frame_records(
        constraints.rows[provenance_columns]
    )
    hard_quotes = [
        dict(quote)
        for quote in forward_manifest["quotes"]
        if str(quote["load_type"]).upper() == "BASE"
    ]
    solver_settings = monthly_solver_settings(config)
    product_policy = {
        "schema_version": "monthly_quote_hierarchy_policy.v1",
        "priority": ["MONTH", "QUARTER", "CALENDAR"],
        "quote_conflict_tolerance_eur_mwh": float(
            solver_settings["quote_conflict_tolerance"]
        ),
        "hard_repricing_tolerance_eur_mwh": float(
            solver_settings["constraint_tolerance"]
        ),
        "stationarity_tolerance": float(solver_settings["stationarity_tolerance"]),
        "residual_lineage_required": True,
    }
    production = staging / "manifests" / "production_monthly_curve_manifest_ch.json"
    monthly_solution = {"2027-01": 83.5, "2027-02": 82.0, "2027-03": 81.0}
    monthly_solution_hash = _sha256_json(monthly_solution)
    production.write_text(
        json.dumps(
            {
                "market": "CH",
                "monthly_level_authority": "solver",
                "solver_config": solver_settings,
                "solver_config_hash": _sha256_json(solver_settings),
                "solver_kkt": {
                    "max_abs_constraint_residual": 1e-12,
                    "stationarity_residual": 1e-12,
                    "condition_number": 100.0,
                    "ridge_used": False,
                    "solved_by_lstsq": False,
                },
                "monthly_solution_hash": monthly_solution_hash,
                "monthly_solution": monthly_solution,
                "active_constraints_hash": _hash_frame(constraints.rows),
                "quote_diagnostics_hash": _hash_frame(constraints.quote_diagnostics),
                "constraint_provenance_rows": constraint_provenance,
                "constraint_provenance_hash": _sha256_json(constraint_provenance),
                "hard_quotes": hard_quotes,
                "hard_quote_set_hash": _sha256_json(hard_quotes),
                "active_config_hash": production_hash or active_hash,
                "promotion_eligible": True,
                "delivery_months": [str(month) for month in delivery_months],
                "forward_snapshot": forward_manifest,
                "forward_eligibility": forward_eligibility.to_manifest(),
                "product_hierarchy_policy": product_policy,
                "product_hierarchy_policy_sha256": _sha256_json(product_policy),
            }
        ),
        encoding="utf-8",
    )
    if governed_data_root is not None:
        production_payload = json.loads(production.read_text(encoding="utf-8"))
        production_payload["source_hashes"] = {
            "eex_forwards_history": history_hash,
            "eex_forwards_history_source_summary": _sha256_json(history_summary),
        }
        production_payload["eex_forwards_history_source_summary"] = history_summary
        production.write_text(json.dumps(production_payload), encoding="utf-8")
    run_manifest = {
        "schema_version": "lt_candidate_run.v1",
        "run_id": "run-001",
        "pipeline_scope": "LT_ONLY",
        "promotion_eligible": False,
        "probabilistic_status": "DETERMINISTIC_ONLY",
        "interval_columns": [],
        "intervals_permitted_for_pricing_or_risk": False,
        "candidate_serialized_at_utc": "2026-07-13T23:59:59+00:00",
        "reference_timestamp": "2026-07-13T23:59:59+00:00",
        "reference_timestamp_is_explicit": True,
        "config_path": config_path.relative_to(staging).as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "pre_run_governance_manifest": pre_run_path.relative_to(staging).as_posix(),
        "pre_run_governance_manifest_sha256": hashlib.sha256(
            pre_run_path.read_bytes()
        ).hexdigest(),
        "data_contract": {
            "archive_path": snapshot.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        },
        "data_pointer": {
            "archive_path": pointer.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
            "logical_path": "views/pfc_lt/current.json",
        },
        "data_generation_id": "input-001",
        "input_sources": input_receipts,
        "source_evidence": {
            "forward_source_archive": source.relative_to(staging).as_posix(),
            "forward_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "forward_quote_snapshot": quotes.relative_to(staging).as_posix(),
            "forward_quote_snapshot_sha256": hashlib.sha256(quotes.read_bytes()).hexdigest(),
            "forward_snapshot_id": forward_snapshot.snapshot_id,
            "forward_observation_id": forward_snapshot.observation_id,
        },
        "markets": {
            "CH": {
                "rows": int(len(pfc_index)),
                "columns": list(pd.read_parquet(pfc_path).columns),
                "probabilistic_status": "DETERMINISTIC_ONLY",
                "pfc_parquet": pfc_path.relative_to(staging).as_posix(),
                "pfc_parquet_sha256": hashlib.sha256(pfc_path.read_bytes()).hexdigest(),
                "monthly_curve_manifest": production.relative_to(staging).as_posix(),
                "monthly_curve_manifest_sha256": hashlib.sha256(
                    production.read_bytes()
                ).hexdigest(),
            }
        },
    }
    deterministic_artifacts = {
        path.relative_to(staging).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root_name in ("markets", "models")
        if (staging / root_name).is_dir()
        for path in sorted((staging / root_name).rglob("*"))
        if path.is_file()
    }
    run_manifest["deterministic_artifacts"] = deterministic_artifacts
    run_manifest["deterministic_artifacts_sha256"] = hashlib.sha256(
        json.dumps(
            deterministic_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if governed_data_root is not None:
        run_manifest["data_layout"] = "external_v2"
        run_manifest["data_root_id"] = "fmv_data_view:pfc_lt"
        run_manifest["freshness_reports"] = freshness_reports
    (staging / "manifests" / "candidate_run_manifest.json").write_text(
        json.dumps(run_manifest),
        encoding="utf-8",
    )
    return staging


def test_assembly_replays_hourly_export_from_staged_pfc(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)

    manifest = assemble_candidate_derived_evidence(staging, run_id="run-001")

    assert manifest["schema_version"] == "lt_candidate_derived_evidence_assembly.v1"
    assert set(manifest["sources"]) == {
        "active_config",
        "input_snapshot_manifest",
        "input_snapshot_pointer",
        "forward_source",
        "eex_acquisition_contract",
        "forward_quote_snapshot",
        "production_manifest",
        "pfc_ch_15min",
        "selected_lambda_decision",
        "historical_thresholds",
    }
    hourly = pd.read_csv(staging / HOURLY_EXPORT)
    assert len(hourly) == len(pd.read_parquet(staging / "markets" / "CH" / "pfc_15min.parquet")) // 4
    assert hourly["price_weighted_mean_eur_mwh"].iloc[0] == 83.5
    monthly_gates = pd.read_csv(staging / "audits" / "monthly_curve_gates.csv")
    assert not monthly_gates.empty
    assert not monthly_gates["status"].eq("CRITICAL").any()
    assert verify_candidate_derived_evidence(
        staging, expected_run_id="run-001"
    ) == manifest


def test_assembly_rejects_interval_columns_even_if_candidate_hash_is_updated(
    tmp_path: Path,
) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path)
    pfc["p10"] = pfc["price_shape"] - 10.0
    pfc["p90"] = pfc["price_shape"] + 10.0
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["columns"] = list(pfc.columns)
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(
        pfc_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="interval columns"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_completed_assembly_is_idempotent(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    first = assemble_candidate_derived_evidence(staging, run_id="run-001")
    first_hash = hashlib.sha256((staging / ASSEMBLY_MANIFEST).read_bytes()).hexdigest()

    second = assemble_candidate_derived_evidence(staging, run_id="run-001")

    assert second == first
    assert hashlib.sha256((staging / ASSEMBLY_MANIFEST).read_bytes()).hexdigest() == first_hash


@pytest.mark.parametrize(
    ("month", "expected_quarter_hours"),
    [("2027-03", 2972), ("2027-10", 2980)],
)
def test_solver_delivery_grid_preserves_spring_and_autumn_dst(
    month: str,
    expected_quarter_hours: int,
) -> None:
    grid = _delivery_quarter_hour_grid(pd.PeriodIndex([month], freq="M"))

    assert len(grid) == expected_quarter_hours
    assert grid.tz is not None
    assert grid.is_unique
    assert (
        grid.to_series().diff().dropna() == pd.Timedelta(minutes=15)
    ).all()


def test_verifier_rejects_hourly_export_even_after_manifest_rehash(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    hourly_path = staging / HOURLY_EXPORT
    hourly = pd.read_csv(hourly_path)
    hourly.loc[0, "price_weighted_mean_eur_mwh"] = 999.0
    hourly.to_csv(hourly_path, index=False, lineterminator="\n")
    manifest_path = staging / ASSEMBLY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["derived"]["hourly_export_ch"]["sha256"] = hashlib.sha256(
        hourly_path.read_bytes()
    ).hexdigest()
    manifest["derived"]["hourly_export_ch"]["size_bytes"] = hourly_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="replay"):
        verify_candidate_derived_evidence(staging, expected_run_id="run-001")


def test_verifier_rejects_forged_export_claims_and_dependency_graph(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    export_path = staging / "manifests" / "hourly_export_manifest_ch.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    export["market"] = "FORGED"
    export["producer_contract"] = "attacker.v1"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    manifest_path = staging / ASSEMBLY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependency_graph"] = {"forged": ["nothing"]}
    manifest["derived"]["export_manifest"]["sha256"] = hashlib.sha256(
        export_path.read_bytes()
    ).hexdigest()
    manifest["derived"]["export_manifest"]["size_bytes"] = export_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        CandidateEvidenceAssemblyError,
        match="dependency_graph|export_manifest.replay",
    ):
        verify_candidate_derived_evidence(staging, expected_run_id="run-001")


def test_assembly_rejects_production_manifest_config_drift(tmp_path: Path) -> None:
    staging = _fixture(tmp_path, production_hash="f" * 64)

    with pytest.raises(CandidateEvidenceAssemblyError, match="hash parity failed"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solved_by_lstsq", True),
        ("ridge_used", True),
        ("stationarity_residual", 1.0),
        ("max_abs_constraint_residual", 1.0),
        ("condition_number", "not-finite"),
        ("condition_number", 1e15),
    ],
)
def test_assembly_rejects_failed_solver_numerical_diagnostics(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    staging = _fixture(tmp_path)
    production_path = (
        staging / "manifests" / "production_monthly_curve_manifest_ch.json"
    )
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production["solver_kkt"][field] = value
    production_path.write_text(json.dumps(production), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["monthly_curve_manifest_sha256"] = hashlib.sha256(
        production_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="numerical diagnostics"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_pfc_monthly_level_drift(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path)
    pfc["price_shape"] += 5.0
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(
        pfc_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="monthly means"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_partial_solver_month_even_after_row_count_update(
    tmp_path: Path,
) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path).iloc[:-4]
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["rows"] = len(pfc)
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(
        pfc_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="exactly cover"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_inflated_repricing_tolerance(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    production_path = (
        staging / "manifests" / "production_monthly_curve_manifest_ch.json"
    )
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production["product_hierarchy_policy"][
        "hard_repricing_tolerance_eur_mwh"
    ] = 6.0
    production["product_hierarchy_policy_sha256"] = _sha256_json(
        production["product_hierarchy_policy"]
    )
    production_path.write_text(json.dumps(production), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["monthly_curve_manifest_sha256"] = hashlib.sha256(
        production_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="policy mismatch"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_duplicated_forward_quote_snapshot(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    quotes_path = staging / "evidence" / "eex" / "forward_quote_snapshot.parquet"
    quotes = pd.read_parquet(quotes_path)
    pd.concat([quotes, quotes.iloc[[0]]], ignore_index=True).to_parquet(
        quotes_path,
        index=False,
    )
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["source_evidence"]["forward_quote_snapshot_sha256"] = hashlib.sha256(
        quotes_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="quote snapshot"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_misaligned_quarter_hour_grid(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    pfc_path = staging / "markets" / "CH" / "pfc_15min.parquet"
    pfc = pd.read_parquet(pfc_path)
    pfc.index = pfc.index + pd.Timedelta(minutes=7)
    pfc.to_parquet(pfc_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["markets"]["CH"]["pfc_parquet_sha256"] = hashlib.sha256(
        pfc_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="quarter-hour aligned"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_missing_governed_input_snapshot(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_contract"] = None
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="input snapshot is missing"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_signed_divergent_input_acquisition_id(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    snapshot_path = staging / "evidence" / "data" / "lt_input_snapshot.json"
    contract = json.loads(snapshot_path.read_text(encoding="utf-8"))
    contract.pop("acquisition_attestation")
    contract["files"]["entso"]["acquisition_id"] = "foreign-acquisition"
    signed = sign_acquisition_contract(
        contract,
        private_key_path=os.environ[
            "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"
        ],
    )
    snapshot_path.write_text(json.dumps(signed), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_contract"]["sha256"] = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="authentication failed"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_signed_input_path_escape(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    snapshot_path = staging / "evidence" / "data" / "lt_input_snapshot.json"
    contract = json.loads(snapshot_path.read_text(encoding="utf-8"))
    contract.pop("acquisition_attestation")
    contract["files"]["entso"]["path"] = "../foreign/entso.parquet"
    snapshot_path.write_text(
        json.dumps(
            sign_acquisition_contract(
                contract,
                private_key_path=os.environ[
                    "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    pointer_path = staging / "evidence" / "data" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["contract_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_contract"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    run["data_pointer"]["sha256"] = hashlib.sha256(pointer_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="authentication failed"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_extra_input_receipt_role(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["input_sources"]["foreign"] = {
        "role": "foreign",
        "logical_path": "foreign.parquet",
        "size_bytes": 1,
        "sha256": "f" * 64,
        "rows": 1,
        "frame_sha256": "e" * 64,
    }
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="receipt roles are not exact"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_assembly_rejects_rehashed_foreign_pointer_generation(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    pointer_path = staging / "evidence" / "data" / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["generation_id"] = "foreign-generation"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data_pointer"]["sha256"] = hashlib.sha256(pointer_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(CandidateEvidenceAssemblyError, match="generation mismatch"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_eex_acquisition_requires_nonempty_common_identity(tmp_path: Path) -> None:
    workbook = tmp_path / "eex.xlsx"
    workbook.write_bytes(b"fixture")
    contract_path = tmp_path / "eex-contract.json"
    contract_path.write_text(
        json.dumps(
            sign_acquisition_contract(
                {
                    "schema_version": "lt_input_snapshot.v1",
                    "layout": "external_v2",
                    "generation_id": "eex-001",
                    "acquisition_id": "",
                    "calibration_eligible": True,
                    "source_class": "GOVERNED_ACQUISITION",
                    "files": {
                        "eex_forward_source": {
                            "path": workbook.name,
                            "size_bytes": workbook.stat().st_size,
                            "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                            "acquisition_id": "",
                            "available_at_utc": "2026-07-12T18:00:00+00:00",
                            "calibration_eligible": True,
                            "source_system": "EEX_MARKET_DATA",
                        }
                    },
                },
                private_key_path=os.environ[
                    "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(CandidateEvidenceError, match="identity is missing"):
        _verified_eex_acquisition(
            workbook,
            contract_path,
            valuation_timestamp="2026-07-13T00:00:00+00:00",
        )


def test_renamed_ompex_source_is_rejected_by_authenticated_provider(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "generic-market-data.xlsx"
    workbook.write_bytes(b"renamed OMPEX benchmark bytes")
    contract_path = tmp_path / "eex-contract.json"
    contract_path.write_text(
        json.dumps(
            sign_acquisition_contract(
                {
                    "schema_version": "lt_input_snapshot.v1",
                    "layout": "external_v2",
                    "generation_id": "eex-ompex-001",
                    "acquisition_id": "eex-ompex-001",
                    "calibration_eligible": True,
                    "source_class": "GOVERNED_ACQUISITION",
                    "files": {
                        "eex_forward_source": {
                            "path": workbook.name,
                            "size_bytes": workbook.stat().st_size,
                            "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
                            "acquisition_id": "eex-ompex-001",
                            "available_at_utc": "2026-07-12T18:00:00+00:00",
                            "calibration_eligible": True,
                            "source_system": "OMPEX_HFC",
                        }
                    },
                },
                private_key_path=os.environ[
                    "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(CandidateEvidenceError, match="source system is not admissible"):
        _verified_eex_acquisition(
            workbook,
            contract_path,
            valuation_timestamp="2026-07-13T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    "section",
    ["model", "forwards", "flavors", "export", "quality"],
)
def test_assembly_rejects_ompex_marker_in_model_consumed_config(
    tmp_path: Path,
    section: str,
) -> None:
    extra = {section: {"prohibited_training_source": "generic_ompex_payload"}}
    if section == "forwards":
        extra[section]["monthly_curve_solver"] = {
            "enabled": True,
            "target_markets": ["CH"],
        }
    if section == "quality":
        extra[section].update(
            {"benchmark_policy": "advisory", "fail_on_benchmark": False}
        )
    staging = _fixture(tmp_path, extra_config=extra)

    with pytest.raises(CandidateEvidenceAssemblyError, match="OMPEX/HFC"):
        assemble_candidate_derived_evidence(staging, run_id="run-001")


def test_product_evidence_requires_then_accepts_exact_signed_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")

    pending = assemble_candidate_product_evidence(staging, run_id="run-001")

    assert pending["status"] == "SOURCE_HIERARCHY_POLICY_REQUIRED"
    assert pending["promotion_eligible"] is False
    assert product_evidence_main(
        ["--staging", str(staging), "--run-id", "run-001"]
    ) == 2
    assert "SOURCE_HIERARCHY_POLICY_REQUIRED" in capsys.readouterr().out
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ[
                    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )

    evidence = assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )

    assert evidence["schema_version"] == "candidate_product_normalization_evidence.v1"
    assert verify_candidate_product_evidence(
        staging,
        expected_run_id="run-001",
    ) == evidence
    summary = json.loads(
        (staging / "manifests" / "product_normalization_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["monthly_candidate_binding_status"] == "BOUND"
    assert summary["source_hierarchy_policy"]["status"] == "ACCEPTED_PRODUCTION_APPROVED"
    assert summary["all_gates_pass"] is True
    assert summary["audit_script"] == "scripts/audit_ch_product_normalization.py"
    assert not list(staging.rglob("*.tmp"))


def test_product_evidence_rejects_unsigned_policy(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "unsigned-policy.json"
    policy_path.write_text(
        json.dumps(pending["required_policy_payload"]),
        encoding="utf-8",
    )

    with pytest.raises(CandidateProductEvidenceError, match="not authentic"):
        assemble_candidate_product_evidence(
            staging,
            run_id="run-001",
            source_hierarchy_policy_path=policy_path,
        )


def test_product_inventory_recovers_exact_missing_sibling(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    first = assemble_candidate_product_evidence(staging, run_id="run-001")
    (staging / CONFLICT_INVENTORY).unlink()

    recovered = assemble_candidate_product_evidence(staging, run_id="run-001")

    assert recovered == first
    assert (staging / CONFLICT_INVENTORY).is_file()


def test_rejected_signed_policy_is_not_staged_and_can_be_retried(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    wrong_payload = dict(pending["required_policy_payload"])
    wrong_payload["input_csv_sha256"] = "f" * 64
    wrong_path = tmp_path / "wrong-signed-policy.json"
    wrong_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                wrong_payload,
                private_key_path=os.environ[
                    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(CandidateProductEvidenceError, match="did not pass"):
        assemble_candidate_product_evidence(
            staging,
            run_id="run-001",
            source_hierarchy_policy_path=wrong_path,
        )
    assert not (staging / STAGED_POLICY).exists()

    correct_path = tmp_path / "correct-signed-policy.json"
    correct_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ[
                    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    evidence = assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=correct_path,
    )
    assert evidence["schema_version"] == "candidate_product_normalization_evidence.v1"


def test_product_evidence_rejects_mutated_conflict_inventory(tmp_path: Path) -> None:
    staging = _fixture(tmp_path)
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ[
                    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )
    inventory_path = staging / CONFLICT_INVENTORY
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["quote_conflict_count"] = 999
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(CandidateProductEvidenceError, match="inventory replay mismatch"):
        verify_candidate_product_evidence(staging, expected_run_id="run-001")


def test_canonical_seal_and_locked_finalizer_replay_both_assemblies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    atomic_promotion._release_root_id(release, create=True)
    staging = _fixture(
        tmp_path,
        staging_path=release / "candidates" / ".run-001.staging",
    )
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ[
                    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )

    seal_assembled_candidate_evidence(staging, run_id="run-001")
    verify_assembled_candidate_evidence(staging, expected_run_id="run-001")
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    interrupted_run = json.loads(run_path.read_text(encoding="utf-8"))
    interrupted_run.pop("candidate_evidence_manifest")
    interrupted_run.pop("candidate_evidence_manifest_sha256")
    run_path.write_text(json.dumps(interrupted_run), encoding="utf-8")
    seal_assembled_candidate_evidence(staging, run_id="run-001")
    verify_assembled_candidate_evidence(staging, expected_run_id="run-001")
    for private_key_env in (
        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(private_key_env)
    assert finalize_candidate_main(
        [
            "--release-root",
            str(release),
            "--failure-root",
            str(tmp_path / "failures"),
            "--run-id",
            "run-001",
            "--source-hierarchy-policy",
            str(policy_path),
        ]
    ) == 0
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["status"] == "CANDIDATE_FINALIZED_NOT_PROMOTED"
    assert len(finalized["candidate_bundle_manifest_sha256"]) == 64
    assert finalize_candidate_main(
        [
            "--release-root",
            str(release),
            "--failure-root",
            str(tmp_path / "failures"),
            "--run-id",
            "run-001",
            "--source-hierarchy-policy",
            str(policy_path),
        ]
    ) == 0
    retried = json.loads(capsys.readouterr().out)
    assert retried == finalized
    bundle = verify_candidate_bundle(release, run_id="run-001")
    recovered_bundle = finalize_assembled_candidate_staging(release, run_id="run-001")
    assert recovered_bundle.manifest_sha256 == bundle.manifest_sha256

    assert bundle.path == (release / "candidates" / "run-001").resolve()
    sealed = verify_assembled_candidate_evidence(
        bundle.path,
        expected_run_id="run-001",
    )
    assert sealed["run_id"] == "run-001"
    source = bundle.path / sealed["artifacts"]["audit_gates"]["path"]
    original_read_bytes = Path.read_bytes
    attack_workflow, attack_evidence = _provision_registration_roots(
        tmp_path,
        workflow_name="attack-workflow",
        evidence_name="attack-evidence",
    )
    monkeypatch.setenv(
        "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
        os.environ["PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH"],
    )
    with monkeypatch.context() as attack:
        attack.setattr(
            governed_release,
            "verify_candidate_bundle",
            lambda *args, **kwargs: bundle,
        )
        attack.setattr(
            governed_release,
            "verify_assembled_candidate_evidence",
            lambda *args, **kwargs: sealed,
        )

        def substitute_capture(path: Path) -> bytes:
            if path.resolve() == source.resolve():
                return b"substituted-after-seal"
            return original_read_bytes(path)

        attack.setattr(Path, "read_bytes", substitute_capture)
        with pytest.raises(GovernedReleaseError, match="role hash mismatch"):
            register_assembled_release_request(
                expected_current_event_id=None,
                release_root=release,
                workflow_root=attack_workflow,
                evidence_root=attack_evidence,
                run_id="run-001",
            )
    workflow, evidence = _provision_registration_roots(
        tmp_path,
        workflow_name="workflow",
        evidence_name="release-evidence",
    )
    with monkeypatch.context() as no_recapture:
        no_recapture.setattr(
            governed_release,
            "_capture_artifacts",
            lambda *args, **kwargs: pytest.fail(
                "assembled registration must not recapture flattened artifacts"
            ),
        )
        request = register_assembled_release_request(
            expected_current_event_id=None,
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id="run-001",
        )
    assert request["registration_contract"] == "assembled_candidate_seal.v1"
    assert set(request["artifacts"]) == {
        "audit_gates",
        "historical_thresholds",
        "production_manifest",
        "export_manifest",
        "selected_config_artifact",
        "product_normalization_summary",
    }


def test_strict_finalizer_quarantines_failed_post_rename_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    staging = _fixture(
        tmp_path,
        staging_path=release / "candidates" / ".run-001.staging",
    )
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ[
                    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )
    seal_assembled_candidate_evidence(staging, run_id="run-001")

    with pytest.raises(PromotionError, match="evidence_contract metadata is reserved"):
        finalize_assembled_candidate_staging(
            release,
            run_id="run-001",
            metadata={"evidence_contract": "generic_candidate.v1"},
        )
    assert staging.is_dir()
    assert not (release / "candidates" / "run-001").exists()

    original_verify = atomic_promotion.verify_assembled_candidate_evidence

    def fail_after_rename(path: str | Path, *, expected_run_id: str) -> dict[str, object]:
        if Path(path).name == "run-001":
            raise RuntimeError("injected post-rename replay failure")
        return original_verify(path, expected_run_id=expected_run_id)

    monkeypatch.setattr(
        atomic_promotion,
        "verify_assembled_candidate_evidence",
        fail_after_rename,
    )

    with pytest.raises(PromotionError, match="quarantined at"):
        finalize_assembled_candidate_staging(release, run_id="run-001")

    assert not (release / "candidates" / "run-001").exists()
    assert len(list((release / "candidates").glob(".run-001.failed-*"))) == 1


def test_assembled_release_runs_real_capstone_from_canonical_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    atomic_promotion._release_root_id(release, create=True)
    data_root = tmp_path / "governed-data"
    staging = _fixture(
        tmp_path,
        staging_path=release / "candidates" / ".run-001.staging",
        governed_data_root=data_root,
    )
    assemble_candidate_derived_evidence(staging, run_id="run-001")
    pending = assemble_candidate_product_evidence(staging, run_id="run-001")
    policy_path = tmp_path / "signed-source-hierarchy-policy.json"
    policy_path.write_text(
        json.dumps(
            sign_quote_conflict_policy(
                pending["required_policy_payload"],
                private_key_path=os.environ[
                    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH"
                ],
            )
        ),
        encoding="utf-8",
    )
    assemble_candidate_product_evidence(
        staging,
        run_id="run-001",
        source_hierarchy_policy_path=policy_path,
    )
    seal_assembled_candidate_evidence(staging, run_id="run-001")

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 7, 14, 0, 1, tzinfo=timezone.utc)
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(atomic_promotion, "datetime", FixedDateTime)
    finalize_assembled_candidate_staging(release, run_id="run-001")
    workflow, evidence = _provision_registration_roots(
        tmp_path,
        workflow_name="workflow",
        evidence_name="release-evidence",
    )
    for private_key_env in (
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
    ):
        monkeypatch.delenv(private_key_env, raising=False)
    monkeypatch.setenv(
        "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
        os.environ["PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH"],
    )
    request = register_assembled_release_request(
        expected_current_event_id=None,
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id="run-001",
    )
    monkeypatch.delenv("PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH")
    receipt_key = Ed25519PrivateKey.generate()
    receipt_private = tmp_path / "receipt-private.pem"
    receipt_public = tmp_path / "receipt-public.pem"
    receipt_private.write_bytes(
        receipt_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    receipt_public.write_bytes(
        receipt_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(receipt_public))
    monkeypatch.delenv("PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.setattr(
        capstone,
        "_promotion_now_utc",
        lambda: pd.Timestamp("2026-07-14T00:05:00Z"),
    )

    audited = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id="run-001",
        request_id=str(request["request_id"]),
        signing_private_key=receipt_private,
    )

    assert audited["approved"] is False
    assert audited["status"] == "REJECTED"
    receipts = list(
        (
            workflow
            / "run-001"
            / "audit-results"
            / str(request["request_id"])[:32]
        ).glob("promotion_receipt.json")
    )
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    governance = {row["gate_id"]: row for row in receipt["governance_gates"]}
    assert all(row["status"] == "PASS" for row in governance.values())
    assert governance["delivered_product_normalization_audit"]["status"] == "PASS"
