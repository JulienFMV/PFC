#!/usr/bin/env python3
"""Compatibility builder facade; canonical CLI is run_governed_lt_release build."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from pfc_shaping.pipeline.atomic_promotion import (
    create_candidate_staging,
)
from pfc_shaping.pipeline.candidate_bundle import write_long_term_candidate
from pfc_shaping.pipeline.candidate_evidence import (
    PRE_RUN_EVIDENCE_MANIFEST,
    capture_pre_run_governance_evidence,
)
from pfc_shaping.pipeline.candidate_evidence_assembler import (
    ASSEMBLY_MANIFEST,
    assemble_candidate_derived_evidence,
)
from pfc_shaping.pipeline.candidate_product_evidence import (
    CONFLICT_INVENTORY,
    assemble_candidate_product_evidence,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    EXIT_GOVERNANCE_PENDING,
    add_build_arguments,
    assert_governed_runtime_source,
    assert_namespace_absolute_paths,
    assert_phase_private_key_scope,
    resolve_build_data_root,
)
from pfc_shaping.pipeline.model_governance_contract import (
    validate_selected_lambda_decision,
)
from pfc_shaping.pipeline.monthly_curve_authority import (
    active_monthly_curve_config_payload,
    monthly_solver_settings,
)
from pfc_shaping.pipeline.production_phases import load_inputs, run_long_term_phase
from pfc_shaping.pipeline.strict_structured_data import load_strict_json


def build_candidate(args: argparse.Namespace) -> dict[str, object]:
    """Build one staged candidate and return its non-promotional status."""

    assert_phase_private_key_scope("build")
    assert_governed_runtime_source(args.source_revision)
    assert_namespace_absolute_paths(
        args,
        (
            "release_root",
            "failure_root",
            "config",
            "input_snapshot_contract",
            "input_pointer_contract",
            "publication_head_observation",
            "data_root",
            "historical_thresholds",
            "historical_thresholds_receipt",
            "selected_lambda_decision",
            "selected_lambda_decision_receipt",
            "eex_report_path",
            "eex_acquisition_contract",
        ),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger("PFC_LT_CANDIDATE")
    staging = create_candidate_staging(args.release_root, run_id=args.run_id)
    pre_run_governance = capture_pre_run_governance_evidence(
        staging,
        run_id=args.run_id,
        valuation_timestamp=pd.Timestamp(args.reference_timestamp).tz_convert("UTC").isoformat(),
        build_timestamp=args.build_timestamp,
        historical_thresholds_path=args.historical_thresholds,
        historical_thresholds_receipt_path=args.historical_thresholds_receipt,
        selected_lambda_decision_path=args.selected_lambda_decision,
        selected_lambda_decision_receipt_path=args.selected_lambda_decision_receipt,
        eex_forward_source_path=args.eex_report_path,
        eex_acquisition_contract_path=args.eex_acquisition_contract,
        runtime_config_path=args.config,
        expected_runtime_config_sha256=args.config_sha256,
        source_revision=args.source_revision,
        input_snapshot_sha256=args.input_snapshot_sha256,
        input_pointer_sha256=args.input_pointer_sha256,
        input_generation_id=args.input_generation_id,
        publication_head_observation_sha256=(
            args.publication_head_observation_sha256
        ),
        publication_head_challenge_nonce=args.publication_head_challenge_nonce,
        peak_source_policy=args.peak_source_policy,
        use_seasonal_hourly_shape=args.use_seasonal_hourly_shape,
    )
    config_entry = pre_run_governance["artifacts"]["runtime_config"]
    staged_config = staging / str(config_entry["path"])
    eex_entry = pre_run_governance["artifacts"]["eex_forward_source"]
    staged_eex = staging / str(eex_entry["path"])
    eex_available_at = eex_entry["acquisition_contract"]["available_at_utc"]
    runtime_root = staged_config.parent
    inputs = load_inputs(
        str(runtime_root),
        logger,
        reference_timestamp=args.reference_timestamp,
        data_root=args.data_root,
        eex_report_path=staged_eex,
        eex_report_available_at=eex_available_at,
        config_path=staged_config,
        expected_config_sha256=args.config_sha256,
        input_snapshot_contract=args.input_snapshot_contract,
        input_pointer_contract=args.input_pointer_contract,
        expected_input_snapshot_sha256=args.input_snapshot_sha256,
        expected_input_pointer_sha256=args.input_pointer_sha256,
        expected_input_generation_id=args.input_generation_id,
        publication_head_observation=args.publication_head_observation,
        expected_publication_head_observation_sha256=(
            args.publication_head_observation_sha256
        ),
        publication_head_challenge_nonce=args.publication_head_challenge_nonce,
    )
    selected_entry = pre_run_governance["artifacts"]["selected_lambda_decision"]
    selected_path = staging / str(selected_entry["path"])
    selected_decision = load_strict_json(selected_path.read_text(encoding="utf-8"))
    validate_selected_lambda_decision(
        selected_decision,
        expected_config=active_monthly_curve_config_payload(
            monthly_solver_settings(inputs.config)
        ),
    )
    long_term = run_long_term_phase(
        project_root=str(runtime_root),
        inputs=inputs,
        peak_source_policy=args.peak_source_policy,
        use_seasonal_hourly_shape=args.use_seasonal_hourly_shape,
        logger=logger,
    )
    run_manifest = write_long_term_candidate(
        staging,
        run_id=args.run_id,
        long_term=long_term,
        inputs=inputs,
        project_root=runtime_root,
        pre_run_governance_manifest=pre_run_governance,
    )
    derived_evidence = assemble_candidate_derived_evidence(
        staging,
        run_id=args.run_id,
    )
    product_evidence = assemble_candidate_product_evidence(
        staging,
        run_id=args.run_id,
    )
    return {
        "schema_version": "governed_lt_build_result.v1",
        "status": "CANDIDATE_STAGED_EVIDENCE_PENDING",
        "state": "CANDIDATE_STAGED_EVIDENCE_PENDING",
        "next_action": product_evidence["status"],
        "run_id": args.run_id,
        "staging_path": str(staging),
        "candidate_run_manifest": str(
            staging / "manifests" / "candidate_run_manifest.json"
        ),
        "pre_run_governance_manifest": str(staging / PRE_RUN_EVIDENCE_MANIFEST),
        "derived_evidence_assembly": str(staging / ASSEMBLY_MANIFEST),
        "derived_evidence_roles": sorted(derived_evidence["derived"]),
        "quote_conflict_inventory": str(staging / CONFLICT_INVENTORY),
        "required_policy_payload": product_evidence["required_policy_payload"],
        "reference_timestamp": run_manifest["reference_timestamp"],
        "governed_run_spec_sha256": pre_run_governance[
            "governed_run_spec_sha256"
        ],
        "promotion_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(json.dumps(build_candidate(args), sort_keys=True))
    return EXIT_GOVERNANCE_PENDING


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_build_arguments(parser)
    args = parser.parse_args(argv)
    assert_phase_private_key_scope("build")
    return resolve_build_data_root(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
