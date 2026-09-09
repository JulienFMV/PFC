"""Superseded v1 policy scaffold for the future CH LT successor.

The canonical v1 verifier is permanently disabled before every artifact read
because its former chain exposed outcome-bearing T057 registry metadata.  Its
exact JSON bytes may be assessed only as a non-executable development-policy
scaffold for v2; they satisfy no evidence slot and authorize nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

SCHEMA_VERSION = "ch_lt_successor_candidate_core.v1"
ASSESSMENT_SCHEMA_VERSION = "ch_lt_successor_candidate_core_assessment.v1"
STATUS = "FROZEN_CANDIDATE_CORE_VALID_NOT_ADMITTED_NOT_EXECUTABLE"
SUPERSEDED_STATUS = "V1_SUPERSEDED_OUTCOME_METADATA_EXPOSED_NOT_ADMITTED"
CORE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-SUCCESSOR-CANDIDATE-CORE-20260730.json"
)
CORE_SHA256 = "87c9447230efce8e5a1c56086d961075621ff521a1f1bee90aa2e1abf97960d9"
CORE_ID = "ab835ce1995377129f990d210bd1ba3490def226ce63e7cde25b57b553d3b4e9"

READINESS_V1_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-SUCCESSOR-READINESS-CONTRACT-20260729.json"
)
PREREGISTRATION_SUPERSESSION_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json"
)
ESTIMAND_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json"
)
COMPUTE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/CH-LT-COMPUTE-RUNTIME-DRAFT-20260727.json"
)
DEPENDENCE_SUPERSESSION_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-DEPENDENCE-POWER-PREFLIGHT-SUPERSESSION-20260724.json"
)
T057_REGISTRY_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/T057-EVIDENCE-SUPERSESSION-REGISTRY.json"
)
MARKET_TIME_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/CH-MARKET-TIME-REGIME-CONTRACT-20260730.json"
)
LITERATURE_RELATIVE_PATH = (
    ".planning/phases/14-lt-audit-remediation/"
    "PFC-2026-LITERATURE-EVIDENCE-AND-GATES-20260717.md"
)

REQUIRED_EVIDENCE = (
    "GOVERNED_POINT_IN_TIME_CH_AND_EEX_EVIDENCE",
    "NATIVE_CH_LAYER_TRUTH_AND_POST_EPISODE_OUTCOMES",
    "EXACT_ORIGIN_TARGET_MASK_AND_INNER_FOLD_INVENTORIES",
    "FROZEN_DETERMINISTIC_CANDIDATE_BASELINE_COMPLEXITY_AND_MARKET_GATES",
    "FROZEN_DEPENDENCE_POWER_AND_MULTIPLICITY_DESIGN",
    "FROZEN_PROBABILISTIC_SCENARIO_AND_MONTE_CARLO_DESIGN",
    "FMV_RISK_APPROVED_ECONOMIC_MATERIALITY",
    "QUALIFIED_CPU_ORACLE_AND_CPU_GPU_RUNTIME_PARITY",
    "INDEPENDENT_SECURITY_ITOPS_AND_QUANT_REVIEWS",
    "EXTERNAL_ADMISSION_ENVELOPE_AND_ONE_SHOT_LEDGER",
)

_BINDING_HASHES = {
    READINESS_V1_RELATIVE_PATH: "734a7824ec747c829526774b346da441b45fff7cff5c9eb79ffc653ac78c7b8e",
    PREREGISTRATION_SUPERSESSION_RELATIVE_PATH: (
        "76dba0b05948d336268b0a50c16df82ecd1c5138626c3fa87ee219148cb8e5e8"
    ),
    ESTIMAND_RELATIVE_PATH: "4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931",
    COMPUTE_RELATIVE_PATH: "b231345e96e7664ae02b7dbf3514af87d47ded7783034eaab1f8d449a28fe96f",
    DEPENDENCE_SUPERSESSION_RELATIVE_PATH: (
        "9fd9cf706c768716a42967962527a49a9f92c7a16d1e60de0b3d910565312b72"
    ),
    T057_REGISTRY_RELATIVE_PATH: (
        "0efec38b768b5e14add6cbc35c9b0cf9f10eb23f2cbf040813e68fe734ca4cf6"
    ),
    MARKET_TIME_RELATIVE_PATH: (
        "71711ae80b64556b8deab88e70581e1c7a0ef7c684d672b2cb550f2058c19c25"
    ),
    LITERATURE_RELATIVE_PATH: (
        "aa67e9e9a54c30f8973533fd1c385917529e1299560b3d779c18d0b6e7589465"
    ),
}
_BINDING_PATH_BY_ROLE = {
    "readiness_contract_v1": READINESS_V1_RELATIVE_PATH,
    "preregistration_supersession": PREREGISTRATION_SUPERSESSION_RELATIVE_PATH,
    "estimand_contract": ESTIMAND_RELATIVE_PATH,
    "compute_contract": COMPUTE_RELATIVE_PATH,
    "dependence_preflight_supersession": DEPENDENCE_SUPERSESSION_RELATIVE_PATH,
    "t057_supersession_metadata": T057_REGISTRY_RELATIVE_PATH,
    "market_time_regime_contract": MARKET_TIME_RELATIVE_PATH,
    "literature_gate": LITERATURE_RELATIVE_PATH,
}
_BINDING_KEYS_BY_ROLE = {
    "readiness_contract_v1": {"path", "sha256", "contract_id"},
    "preregistration_supersession": {"path", "sha256"},
    "estimand_contract": {"path", "sha256", "contract_id"},
    "compute_contract": {"path", "sha256", "contract_id"},
    "dependence_preflight_supersession": {"path", "sha256"},
    "t057_supersession_metadata": {"path", "sha256", "access"},
    "market_time_regime_contract": {"path", "sha256", "contract_id"},
    "literature_gate": {"path", "sha256"},
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "core_name",
    "lifecycle",
    "structural_bindings",
    "information_firewall",
    "candidate_and_baseline_graphs",
    "origin_fold_and_truth_design",
    "deterministic_market_and_complexity_gates",
    "dependence_power_and_multiplicity_design",
    "probabilistic_scenario_and_monte_carlo_design",
    "economic_decision_design",
    "compute_runtime_design",
    "required_evidence_slots",
    "admission_boundary",
    "scientific_admission",
    "execution_authorized",
    "production_authorization",
    "promotion_gate",
    "future_holdout_consumed",
    "t057_consumed",
    "core_id",
}
_SECTION_SHA256 = {
    "lifecycle": "41f83022106fd5aebc483f20148e201d9b6e5334adbb9c43a17568a24e320153",
    "structural_bindings": "fb7362ff3399c19bf5bdd7efa8eb4a9ecec35da898abd4fe272aacf08e30be93",
    "information_firewall": "739d2f269bd21979e505676bb21e1a2fb3d239cdb61b7658b30cf67a6fad8fdc",
    "candidate_and_baseline_graphs": (
        "6c05bc76286eff432483b0b9a223abf8d593cc638baac5f0aaebc11625c22443"
    ),
    "origin_fold_and_truth_design": (
        "8d27db2b7f80e58ef7f324e05b5a7f6635c4b960504fc9f9f32d7f09dcb6a156"
    ),
    "deterministic_market_and_complexity_gates": (
        "ecea2476f1e12b78a57a7666602aac407a4b788f7490922ee82752b56a3fd019"
    ),
    "dependence_power_and_multiplicity_design": (
        "eb8870e45a0ec5a2630dee861748a78892b55cf8536aa3b5a089617ecb67b876"
    ),
    "probabilistic_scenario_and_monte_carlo_design": (
        "faa71dd290deb51927b0f28862f92c06c68277fe9fc6dcf15a5b743d89446345"
    ),
    "economic_decision_design": (
        "d4c74dfc160a99691bb0b7067cf85e5539d8d62f2352cc6f707eb2e4fb405c77"
    ),
    "compute_runtime_design": (
        "1cf543f4ce51d87f5218040bb7a2468f7999acdc0a46f77673dc290fbc6c0303"
    ),
    "required_evidence_slots": (
        "069b09cafb8c11d18971e9a114af04733a6ecef6aedec0947b3e0c3ba4b3139f"
    ),
    "admission_boundary": (
        "f4a512e1202b1f8372712cfe8d8ef4d7b342d18f7f67c5c732e481bb1fe90bee"
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BYTES = 2_000_000


class ChLtSuccessorCandidateCoreError(ValueError):
    """Raised when the candidate core or one of its bindings is invalid."""


def compute_core_id(document: Mapping[str, object]) -> str:
    """Return the semantic SHA-256 after removing the self-binding field."""

    core = dict(document)
    core.pop("core_id", None)
    payload = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_candidate_core(
    *,
    repo_root: str | Path,
    core_path: str | Path,
    expected_core_sha256: str,
) -> dict[str, object]:
    """Verify the canonical core and only its eight structural bindings."""

    _require_sha256(expected_core_sha256, label="expected core")
    if expected_core_sha256 != CORE_SHA256:
        raise ChLtSuccessorCandidateCoreError(
            "caller-held core SHA-256 does not match frozen canonical identity"
        )
    raise ChLtSuccessorCandidateCoreError(
        "candidate core v1 is superseded: outcome-bearing T057 registry metadata was "
        "exposed; use the outcome-blind v2 core"
    )
    root = assert_absolute_path_has_no_links(_absolute(repo_root))
    if not root.is_dir():
        raise ChLtSuccessorCandidateCoreError("repository root is unavailable")
    canonical_core = _repo_path(root, CORE_RELATIVE_PATH)
    if _path_key(_absolute(core_path)) != _path_key(canonical_core):
        raise ChLtSuccessorCandidateCoreError("candidate core path is not canonical")

    expected = {CORE_RELATIVE_PATH: CORE_SHA256, **_BINDING_HASHES}
    payloads: dict[str, bytes] = {}
    paths: list[Path] = []
    for relative_path, expected_sha256 in expected.items():
        path = _repo_path(root, relative_path)
        payload = read_stable_single_link_file(
            path,
            label=f"CH LT successor candidate core artifact {relative_path}",
            max_bytes=_MAX_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ChLtSuccessorCandidateCoreError(
                f"artifact SHA-256 mismatch: {relative_path}"
            )
        payloads[relative_path] = payload
        paths.append(path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if os.path.samefile(left, right):
                raise ChLtSuccessorCandidateCoreError(
                    "candidate core roles must be physically distinct files"
                )

    document = _strict_mapping(payloads[CORE_RELATIVE_PATH], label="candidate core")
    result = assess_candidate_core(document, document_bytes=payloads[CORE_RELATIVE_PATH])
    _verify_bound_documents(payloads)
    return result


def assess_candidate_core(
    document: Mapping[str, object],
    *,
    document_bytes: bytes,
) -> dict[str, object]:
    """Validate the frozen core while keeping every evidence slot missing."""

    parsed = _strict_mapping(document_bytes, label="candidate core bytes")
    _equal(parsed, document, "candidate core mapping/bytes")
    if hashlib.sha256(document_bytes).hexdigest() != CORE_SHA256:
        raise ChLtSuccessorCandidateCoreError("candidate core document SHA-256 mismatch")
    _exact_keys(document, _TOP_LEVEL_KEYS, label="candidate core")
    _verify_section_hashes(document)
    _equal(document.get("schema_version"), SCHEMA_VERSION, "schema version")
    _equal(
        document.get("core_name"),
        "ch_lt_successor_candidate_core_20260730_v1",
        "core name",
    )
    _equal(document.get("core_id"), CORE_ID, "core id")
    _equal(compute_core_id(document), CORE_ID, "computed core id")
    _verify_lifecycle_and_firewall(document)
    _verify_bindings(document)
    _verify_scientific_design(document)
    _equal(
        document.get("required_evidence_slots"),
        [
            {"requirement": item, "status": "MISSING", "path": None, "sha256": None}
            for item in REQUIRED_EVIDENCE
        ],
        "required evidence slots",
    )
    _equal(
        document.get("admission_boundary"),
        {
            "candidate_core_is_evidence_slot_satisfaction": False,
            "distinct_hash_bound_evidence_required_for_every_slot": True,
            "independent_signed_admission_envelope_required": True,
            "builder_inaccessible_external_cas_worm_and_fresh_head_required": True,
            "predictions_and_scenarios_sealed_before_truth_open": True,
            "monotone_one_shot_attempt_ledger_required": True,
            "reviews_bind_core_and_all_evidence_hashes": True,
        },
        "admission boundary",
    )
    for field in (
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
        "future_holdout_consumed",
        "t057_consumed",
    ):
        _equal(document.get(field), False, field)
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "status": STATUS,
        "core_path": CORE_RELATIVE_PATH,
        "core_sha256": CORE_SHA256,
        "core_id": CORE_ID,
        "candidate_core_authored": True,
        "candidate_core_frozen": True,
        "candidate_core_admitted": False,
        "successor_exists": False,
        "readiness_complete": False,
        "missing_evidence": list(REQUIRED_EVIDENCE),
        "future_holdout_consumed": False,
        "t057_consumed": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def _verify_lifecycle_and_firewall(document: Mapping[str, object]) -> None:
    _equal(
        document.get("lifecycle"),
        {
            "state": "FROZEN_RECEIPT_FREE_CORE_NOT_ADMITTED",
            "frozen_at_utc": None,
            "trusted_time_receipt_sha256": None,
            "receipt_free": True,
            "truth_free": True,
            "candidate_core_is_admitted_successor": False,
            "candidate_core_can_execute": False,
            "candidate_core_can_open_truth": False,
            "post_freeze_change_requires_new_core": True,
        },
        "lifecycle",
    )
    _equal(
        document.get("information_firewall"),
        {
            "allowed_t057_access": "SUPERSESSION_REGISTRY_METADATA_ONLY",
            "referenced_t057_artifact_reads": "FORBIDDEN",
            "future_holdout_truth_reads": "FORBIDDEN",
            "post_origin_truth_reads_during_fit_selection_generation": "FORBIDDEN",
            "holdout_derived_hyperparameters_thresholds_or_graphs": "FORBIDDEN",
            "superseded_de_lu_pilot_required_n_reuse": "FORBIDDEN",
            "ompex_role": "BENCHMARK_ONLY_FORBIDDEN_AS_INPUT_TRUTH_LEVEL_OR_MDE_EVIDENCE",
            "future_holdout_consumed": False,
            "t057_consumed": False,
        },
        "information firewall",
    )


def _verify_bindings(document: Mapping[str, object]) -> None:
    bindings = _mapping(document.get("structural_bindings"), label="bindings")
    expected_roles = set(_BINDING_PATH_BY_ROLE)
    _exact_keys(bindings, expected_roles, label="bindings")
    observed_paths: set[str] = set()
    for role, value in bindings.items():
        binding = _mapping(value, label=f"binding {role}")
        _exact_keys(binding, _BINDING_KEYS_BY_ROLE[role], label=f"binding {role}")
        path = binding.get("path")
        sha256 = binding.get("sha256")
        if not isinstance(path, str):
            raise ChLtSuccessorCandidateCoreError(f"binding {role} path is invalid")
        _equal(path, _BINDING_PATH_BY_ROLE[role], f"binding {role} path")
        _equal(sha256, _BINDING_HASHES[path], f"binding {role} SHA-256")
        if path in observed_paths:
            raise ChLtSuccessorCandidateCoreError("binding paths must be distinct")
        observed_paths.add(path)
    _equal(
        _mapping(bindings["readiness_contract_v1"], label="readiness binding").get(
            "contract_id"
        ),
        "5e655cab1ca090cd067100dc8eb06161811b77991132e7c299883a7eb249e706",
        "readiness contract id",
    )
    _equal(
        _mapping(bindings["estimand_contract"], label="estimand binding").get(
            "contract_id"
        ),
        "da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344",
        "estimand contract id",
    )
    _equal(
        _mapping(bindings["compute_contract"], label="compute binding").get(
            "contract_id"
        ),
        "d06710ba8ebee2364b81930fce51d17768f206521edfe62336ff2abdef60930a",
        "compute contract id",
    )
    _equal(
        _mapping(bindings["market_time_regime_contract"], label="market time binding").get(
            "contract_id"
        ),
        "898d35e37a2df9dc9814039698eb605fdf220ece33d2035c7c7a2ea1b7bc9dba",
        "market time contract id",
    )
    _equal(
        _mapping(bindings["t057_supersession_metadata"], label="T057 binding").get(
            "access"
        ),
        "REGISTRY_METADATA_ONLY_NO_REFERENCED_ARTIFACT_READ",
        "T057 metadata access",
    )


def _verify_scientific_design(document: Mapping[str, object]) -> None:
    graphs = _mapping(
        document.get("candidate_and_baseline_graphs"), label="candidate graphs"
    )
    _exact_keys(
        graphs,
        {
            "manifest_schema",
            "shared_level_graph",
            "monthly_level_authority",
            "candidate_graphs",
            "baseline_graphs",
            "feature_groups",
            "neighbor_information_role",
            "candidate_graph_and_feature_bytes_required_before_execution",
            "selection_scope",
            "selection_rule",
            "post_selection_month_patch",
            "lt_imports_ct",
        },
        label="candidate graphs",
    )
    _equal(graphs.get("monthly_level_authority"), "SOLVER", "monthly authority")
    _equal(graphs.get("neighbor_information_role"), "ZERO_MEAN_SHAPE_ONLY", "neighbor role")
    _equal(graphs.get("post_selection_month_patch"), "FORBIDDEN", "month patch")
    _equal(graphs.get("lt_imports_ct"), False, "LT/CT separation")
    _equal(
        graphs.get("selection_rule"),
        "ONE_STANDARD_ERROR_THEN_LOWEST_EFFECTIVE_COMPLEXITY_SUBJECT_TO_ALL_HARD_GATES",
        "complexity selection rule",
    )
    candidates = graphs.get("candidate_graphs")
    if not isinstance(candidates, list):
        raise ChLtSuccessorCandidateCoreError("candidate graph inventory is invalid")
    candidate_ids: list[object] = []
    for index, item in enumerate(candidates):
        candidate = _mapping(item, label=f"candidate graph {index}")
        _exact_keys(
            candidate,
            {
                "candidate_id",
                "hourly_shape_family",
                "intraday_shape_family",
                "probabilistic_family",
                "negative_price_support",
            },
            label=f"candidate graph {index}",
        )
        _equal(candidate.get("negative_price_support"), True, "negative-price support")
        candidate_ids.append(candidate.get("candidate_id"))
    _equal(
        candidate_ids,
        [
            "ROBUST_REGULARIZED_LINEAR_GAM_SHAPE_V1",
            "MONOTONE_HIST_GRADIENT_BOOSTING_SHAPE_V1",
        ],
        "candidate graph inventory",
    )
    _equal(
        candidates,
        [
            {
                "candidate_id": "ROBUST_REGULARIZED_LINEAR_GAM_SHAPE_V1",
                "hourly_shape_family": "ROBUST_REGULARIZED_LINEAR_PLUS_CYCLIC_SPLINE",
                "intraday_shape_family": (
                    "NATIVE_PRODUCT_RESIDUAL_ONLY_OTHERWISE_UNSUPPORTED"
                ),
                "probabilistic_family": (
                    "JOINT_BLOCK_RESIDUAL_SCENARIOS_WITH_CONDITIONAL_SCALE"
                ),
                "negative_price_support": True,
            },
            {
                "candidate_id": "MONOTONE_HIST_GRADIENT_BOOSTING_SHAPE_V1",
                "hourly_shape_family": (
                    "HIST_GRADIENT_BOOSTING_WITH_FROZEN_MONOTONIC_CONSTRAINTS"
                ),
                "intraday_shape_family": (
                    "NATIVE_PRODUCT_RESIDUAL_ONLY_OTHERWISE_UNSUPPORTED"
                ),
                "probabilistic_family": (
                    "JOINT_BLOCK_RESIDUAL_SCENARIOS_WITH_CONDITIONAL_SCALE"
                ),
                "negative_price_support": True,
            },
        ],
        "candidate graph definitions",
    )
    _equal(
        graphs.get("baseline_graphs"),
        ["MARKET_CONSTRAINED_SEASONAL_NAIVE_V1", "INCUMBENT_CH_LT_FROZEN_V1"],
        "baseline graph inventory",
    )

    origin = _mapping(
        document.get("origin_fold_and_truth_design"), label="origin/fold/truth design"
    )
    _exact_keys(
        origin,
        {
            "market",
            "timezone",
            "delivery_horizon",
            "model_output_grid_minutes",
            "pre_transition_native_truth_minutes",
            "post_transition_native_truth_minutes",
            "transition_status",
            "origin_unit",
            "outer_origin_cadence",
            "same_origins_targets_masks_for_all_models",
            "inner_selection",
            "minimum_inner_origins",
            "minimum_unique_outer_origins_floor",
            "minimum_unique_outer_origins_final",
            "minimum_per_season",
            "minimum_per_predeclared_stress_regime",
            "overlapping_origins_are_independent",
            "direct_ch_native_truth_required",
            "hourly_duplication_as_native_quarter_hour_truth",
            "future_episode_policy",
            "future_episode_count",
            "future_episode_length_days",
            "exact_origin_target_mask_inner_fold_and_truth_manifests_required",
        },
        label="origin/fold/truth design",
    )
    _equal(origin.get("market"), "CH", "origin market")
    _equal(origin.get("delivery_horizon"), "M01_M36_FULL_LOCAL_DELIVERY_MONTHS", "horizon")
    _equal(origin.get("same_origins_targets_masks_for_all_models"), True, "common rows")
    _equal(origin.get("overlapping_origins_are_independent"), False, "origin dependence")
    _equal(
        origin.get("hourly_duplication_as_native_quarter_hour_truth"),
        "FORBIDDEN",
        "quarter-hour truth",
    )
    _equal(origin.get("future_episode_count"), 4, "future episode count")

    gates = _mapping(
        document.get("deterministic_market_and_complexity_gates"),
        label="deterministic gates",
    )
    _exact_keys(
        gates,
        {
            "primary_metric",
            "secondary_metrics",
            "required_repricing_products",
            "monthly_solver_repricing_tolerance_eur_mwh",
            "final_curve_repricing_tolerance_eur_mwh",
            "monthly_level_preservation_tolerance_eur_mwh",
            "cascade_invariance_tolerance_eur_mwh",
            "quote_curve_jacobian_required",
            "jacobian_rank_condition_and_perturbation_stability_required",
            "maximum_condition_number",
            "hard_market_gates_compensable_by_shape_or_economic_scores",
            "unsupported_or_unstable_complexity_can_pass",
            "all_hard_gates_cpu_float64_only",
        },
        label="deterministic gates",
    )
    _equal(gates.get("monthly_solver_repricing_tolerance_eur_mwh"), 1e-9, "solver tolerance")
    _equal(gates.get("final_curve_repricing_tolerance_eur_mwh"), 1e-6, "final tolerance")
    _equal(gates.get("cascade_invariance_tolerance_eur_mwh"), 1e-8, "cascade tolerance")
    _equal(gates.get("all_hard_gates_cpu_float64_only"), True, "CPU hard gates")
    _equal(
        gates.get("hard_market_gates_compensable_by_shape_or_economic_scores"),
        False,
        "hard gate compensation",
    )

    dependence = _mapping(
        document.get("dependence_power_and_multiplicity_design"),
        label="dependence/power design",
    )
    _exact_keys(
        dependence,
        {
            "confirmatory_unit",
            "dependence_cluster",
            "cross_fitting_creates_independent_units",
            "block_method",
            "block_length_rule",
            "bootstrap_replicates",
            "bootstrap_rng",
            "bootstrap_seed",
            "hac_cross_check",
            "familywise_alpha",
            "target_power",
            "confirmatory_family",
            "multiplicity_control",
            "simultaneous_confidence_intervals_required",
            "effect_variance_source",
            "mde_rule",
            "plugin_or_leave_one_out_required_n_as_confidence_bound",
            "insufficient_effective_clusters_or_power",
        },
        label="dependence/power design",
    )
    _equal(dependence.get("confirmatory_unit"), "UNIQUE_OUTER_ORIGIN", "inference unit")
    _equal(dependence.get("cross_fitting_creates_independent_units"), False, "cross-fitting")
    _equal(dependence.get("familywise_alpha"), 0.05, "familywise alpha")
    _equal(dependence.get("target_power"), 0.8, "target power")
    _equal(dependence.get("multiplicity_control"), "HOLM_FAMILYWISE_ERROR", "multiplicity")
    _equal(
        dependence.get("plugin_or_leave_one_out_required_n_as_confidence_bound"),
        "FORBIDDEN",
        "plugin required N",
    )

    scenarios = _mapping(
        document.get("probabilistic_scenario_and_monte_carlo_design"),
        label="probabilistic/MC design",
    )
    _exact_keys(
        scenarios,
        {
            "primary_representation",
            "quantile_grid",
            "scenario_monthly_policy",
            "scenario_specific_monthly_risk",
            "pathwise_fixed_monthly_level",
            "scenario_count_rule",
            "scenario_count",
            "rng_engine",
            "seed_derivation",
            "backend",
            "chunk_and_generation_order",
            "common_random_numbers",
            "marginal_scores",
            "multivariate_scores",
            "energy_score_dimension_strategy",
            "variogram_order",
            "variogram_lags_quarter_hours",
            "variogram_weights",
            "calibration_slices",
            "tail_events",
            "mc_error_gate",
            "insufficient_mc_precision_or_calibration_cells",
            "post_truth_count_seed_chunk_score_or_threshold_change",
        },
        label="probabilistic/MC design",
    )
    _equal(
        scenarios.get("scenario_monthly_policy"),
        "ENERGY_WEIGHTED_ENSEMBLE_EXPECTATION_EQUALS_MONTHLY_SOLVER_FORWARD",
        "scenario monthly policy",
    )
    _equal(scenarios.get("scenario_count"), None, "scenario count pending study")
    _equal(
        scenarios.get("scenario_count_rule"),
        "SMALLEST_POWER_OF_TWO_4096_TO_65536_PASSING_PREFROZEN_MC_ERROR_GATE",
        "scenario count rule",
    )
    _equal(scenarios.get("rng_engine"), "NUMPY_PCG64_CPU_ONLY", "scenario RNG")
    _equal(scenarios.get("common_random_numbers"), True, "common random numbers")
    _equal(
        scenarios.get("post_truth_count_seed_chunk_score_or_threshold_change"),
        "FORBIDDEN_NEW_CORE_REQUIRED",
        "post-truth scenario change",
    )

    economic = _mapping(document.get("economic_decision_design"), label="economic design")
    _exact_keys(
        economic,
        {
            "required_populations",
            "fil_and_acc_basis",
            "bloc_13_basis",
            "hydro_basis",
            "required_metrics",
            "decision_rule",
            "materiality_thresholds",
            "fmv_risk_signed_decision_sha256",
            "default_capture_premium_as_evidence",
            "ompex_as_mde_or_population_truth",
        },
        label="economic design",
    )
    _equal(
        economic.get("required_populations"),
        ["FIL", "ACC", "BLOC_13", "HYDRO_DISPATCH"],
        "economic populations",
    )
    _equal(economic.get("materiality_thresholds"), None, "materiality pending")
    _equal(economic.get("fmv_risk_signed_decision_sha256"), None, "Risk approval pending")
    _equal(economic.get("default_capture_premium_as_evidence"), "FORBIDDEN", "capture premium")

    runtime = _mapping(document.get("compute_runtime_design"), label="runtime design")
    _exact_keys(
        runtime,
        {
            "cpu_float64_oracle",
            "gpu_allowed",
            "gpu_fit_or_model_selection",
            "minimum_clean_process_repeats_per_backend",
            "tf32_amp_compile",
            "deterministic_algorithms",
            "cpu_gpu_shock_bytes_identical",
            "fallback",
            "exact_wheel_lock_runtime_input_code_and_shock_bindings_required",
        },
        label="runtime design",
    )
    _equal(
        runtime.get("cpu_float64_oracle"),
        "CANONICAL_FOR_ALL_SCORES_GATES_REPRICING_AND_PROJECTIONS",
        "CPU oracle",
    )
    _equal(runtime.get("gpu_fit_or_model_selection"), "FORBIDDEN_V1", "GPU fit")
    _equal(runtime.get("minimum_clean_process_repeats_per_backend"), 3, "runtime repeats")


def _verify_bound_documents(payloads: Mapping[str, bytes]) -> None:
    readiness = _strict_mapping(
        payloads[READINESS_V1_RELATIVE_PATH], label="readiness v1 binding"
    )
    prereg = _strict_mapping(
        payloads[PREREGISTRATION_SUPERSESSION_RELATIVE_PATH],
        label="preregistration supersession binding",
    )
    estimand = _strict_mapping(payloads[ESTIMAND_RELATIVE_PATH], label="estimand binding")
    compute = _strict_mapping(payloads[COMPUTE_RELATIVE_PATH], label="compute binding")
    dependence = _strict_mapping(
        payloads[DEPENDENCE_SUPERSESSION_RELATIVE_PATH],
        label="dependence supersession binding",
    )
    t057 = _strict_mapping(payloads[T057_REGISTRY_RELATIVE_PATH], label="T057 registry")
    market_time = _strict_mapping(
        payloads[MARKET_TIME_RELATIVE_PATH], label="market-time binding"
    )
    _equal(
        readiness.get("contract_id"),
        "5e655cab1ca090cd067100dc8eb06161811b77991132e7c299883a7eb249e706",
        "readiness binding id",
    )
    _equal(_mapping(prereg.get("successor"), label="successor").get("state"), "NOT_CREATED", "successor state")
    _equal(estimand.get("contract_id"), "da4090073a4566f662e47fa59e206e1485d683305a1134f2089ebb13a4daa344", "estimand id")
    _equal(compute.get("contract_id"), "d06710ba8ebee2364b81930fce51d17768f206521edfe62336ff2abdef60930a", "compute id")
    for label, item in (("dependence", dependence), ("T057", t057)):
        _equal(item.get("production_authorization"), False, f"{label} production")
        _equal(item.get("promotion_gate"), False, f"{label} promotion")
    _equal(
        _mapping(market_time.get("execution_controls"), label="market-time controls").get(
            "contract_authorizes_holdout_consumption"
        ),
        False,
        "market-time holdout authority",
    )


def _strict_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        parsed = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChLtSuccessorCandidateCoreError(f"{label} is invalid JSON") from exc
    return _mapping(parsed, label=label)


def _verify_section_hashes(document: Mapping[str, object]) -> None:
    for section, expected_sha256 in _SECTION_SHA256.items():
        payload = json.dumps(
            document.get(section),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ChLtSuccessorCandidateCoreError(
                f"candidate core semantic section mismatch: {section}"
            )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ChLtSuccessorCandidateCoreError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ChLtSuccessorCandidateCoreError(f"{label} keys are not exact")


def _equal(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ChLtSuccessorCandidateCoreError(f"{label} is invalid")


def _repo_path(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative_path:
        raise ChLtSuccessorCandidateCoreError("relative path is unsafe")
    return root.joinpath(*pure.parts)


def _absolute(path: str | Path) -> Path:
    lexical = Path(path).expanduser()
    return lexical if lexical.is_absolute() else Path(os.path.abspath(lexical))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChLtSuccessorCandidateCoreError(f"{label} SHA-256 is invalid")


__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "CORE_ID",
    "CORE_RELATIVE_PATH",
    "CORE_SHA256",
    "REQUIRED_EVIDENCE",
    "STATUS",
    "ChLtSuccessorCandidateCoreError",
    "assess_candidate_core",
    "compute_core_id",
    "verify_candidate_core",
]
