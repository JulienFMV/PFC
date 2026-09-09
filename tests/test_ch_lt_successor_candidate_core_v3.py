from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_successor_candidate_core as packaged_cli
import pfc_shaping.validation.ch_lt_successor_candidate_core_v2 as candidate_core_v2
import pfc_shaping.validation.ch_lt_successor_candidate_core_v3 as candidate_core
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_successor_candidate_core_v3 import (
    CORE_ID,
    CORE_RELATIVE_PATH,
    CORE_SHA256,
    REQUIRED_EVIDENCE,
    STATUS,
    ChLtSuccessorCandidateCoreV3Error,
    assess_candidate_core_v3,
    compute_core_id,
    derive_monthly_overlap_geometry,
    verify_candidate_core_v3,
)
from scripts import audit_ch_lt_successor_candidate_core as audit_script

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.joinpath(*CORE_RELATIVE_PATH.split("/"))


def _document() -> dict[str, object]:
    return json.loads(CORE.read_text(encoding="utf-8"))


def _mutated_bytes(
    document: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    mutated_id = compute_core_id(document)
    document["core_id"] = mutated_id
    monkeypatch.setattr(candidate_core, "CORE_ID", mutated_id)
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    monkeypatch.setattr(candidate_core, "CORE_SHA256", hashlib.sha256(payload).hexdigest())
    return payload


def test_canonical_v3_core_is_contrast_aware_outcome_blind_and_no_go() -> None:
    result = verify_candidate_core_v3(
        repo_root=ROOT,
        core_path=CORE,
        expected_core_sha256=CORE_SHA256,
    )

    assert hashlib.sha256(CORE.read_bytes()).hexdigest() == CORE_SHA256
    assert compute_core_id(_document()) == CORE_ID
    assert result["status"] == STATUS
    assert result["core_id"] == CORE_ID
    assert result["superseded_core_versions"] == ["v1", "v2"]
    assert result["candidate_core_hash_closed_locally"] is True
    assert result["candidate_core_frozen"] is False
    assert result["candidate_core_externally_frozen"] is False
    assert result["candidate_core_admitted"] is False
    assert result["successor_exists"] is False
    assert result["missing_evidence"] == list(REQUIRED_EVIDENCE)
    assert [
        (
            item["maximum_mechanical_overlap_lag_origins"],
            item["mechanical_expected_mean_block_lower_bound_origins"],
        )
        for item in result["contrast_overlap_geometry"]
    ] == [(0, 1), (5, 6), (5, 6), (11, 12), (11, 12), (35, 36)]
    assert result["contrast_overlap_geometry"][-1]["primary_confirmatory_eligible"] is False
    simulation = _document()["overlap_aware_dependence_and_power"]["power_simulation"]
    assert simulation["candidate_origin_counts"].startswith("COMMON_ASCENDING")
    assert "LEAST_FAVOURABLE_NULL_BOUNDARY" in simulation[
        "direction_oriented_null_boundary_rule"
    ]
    assert "COMPLETE_FMV_GATEKEEPING_DECISION" in simulation[
        "alternative_power_calibration"
    ]
    assert simulation["minimum_marginal_power_floor_by_contrast"] is None
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
    ):
        assert result[field] is False


def test_v3_chain_reads_only_the_outcome_blind_t057_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v3_reader = candidate_core.read_stable_single_link_file
    v2_reader = candidate_core_v2.read_stable_single_link_file
    observed: list[str] = []

    def _v3(path: Path, **kwargs: object) -> bytes:
        observed.append(path.as_posix())
        return v3_reader(path, **kwargs)

    def _v2(path: Path, **kwargs: object) -> bytes:
        observed.append(path.as_posix())
        return v2_reader(path, **kwargs)

    monkeypatch.setattr(candidate_core, "read_stable_single_link_file", _v3)
    monkeypatch.setattr(candidate_core_v2, "read_stable_single_link_file", _v2)
    verify_candidate_core_v3(
        repo_root=ROOT,
        core_path=CORE,
        expected_core_sha256=CORE_SHA256,
    )

    assert len(observed) == 11
    t057_reads = [path for path in observed if "T057" in path.upper()]
    assert len(t057_reads) == 1
    assert t057_reads[0].endswith("T057-OUTCOME-BLIND-TOMBSTONE-20260730.json")
    assert all("T057-EVIDENCE-SUPERSESSION-REGISTRY.json" not in path for path in observed)


@pytest.mark.parametrize(
    ("minimum", "maximum", "expected"),
    [
        (1, 1, (1, 0, 1)),
        (1, 6, (6, 5, 6)),
        (7, 12, (6, 5, 6)),
        (13, 24, (12, 11, 12)),
        (25, 36, (12, 11, 12)),
        (1, 36, (36, 35, 36)),
    ],
)
def test_monthly_overlap_geometry_is_exact(
    minimum: int, maximum: int, expected: tuple[int, int, int]
) -> None:
    result = derive_monthly_overlap_geometry(
        minimum_lead_month=minimum, maximum_lead_month=maximum
    )

    assert tuple(result.values()) == expected


@pytest.mark.parametrize("minimum,maximum", [(0, 1), (1, 0), (True, 1)])
def test_invalid_monthly_overlap_geometry_fails_closed(
    minimum: int, maximum: int
) -> None:
    with pytest.raises(ChLtSuccessorCandidateCoreV3Error):
        derive_monthly_overlap_geometry(
            minimum_lead_month=minimum, maximum_lead_month=maximum
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("lifecycle", "externally_frozen", True),
        ("outcome_blind_firewall", "outcome_bearing_registry_bytes", "ALLOWED"),
        (
            "overlap_aware_dependence_and_power",
            "common_global_mechanical_block_lower_bound_for_all_contrasts",
            36,
        ),
        (
            "overlap_aware_dependence_and_power",
            "stationary_bootstrap_block_parameter",
            "FIXED_MINIMUM_BLOCK_SIZE",
        ),
        (
            "overlap_aware_dependence_and_power",
            "power_simulation",
            {"candidate_origin_counts": "PER_CONTRAST_MULTIPLES"},
        ),
        ("monte_carlo_design_correction", "zero_margin_noninferiority_rule", "ZERO_WIDTH"),
    ],
)
def test_rehashed_v3_policy_mutations_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    document = _document()
    policy = document[section]
    assert isinstance(policy, dict)
    policy[field] = replacement
    payload = _mutated_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorCandidateCoreV3Error, match="semantic section mismatch"):
        assess_candidate_core_v3(document, document_bytes=payload)


def test_rehashed_bucket_geometry_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    policy = document["overlap_aware_dependence_and_power"]
    assert isinstance(policy, dict)
    geometry = policy["mechanical_overlap_geometry"]
    assert isinstance(geometry, list)
    row = geometry[1]
    assert isinstance(row, dict)
    row["mechanical_expected_mean_block_lower_bound_origins"] = 36
    payload = _mutated_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorCandidateCoreV3Error, match="semantic section mismatch"):
        assess_candidate_core_v3(document, document_bytes=payload)


def test_wrong_caller_hash_and_noncanonical_path_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ChLtSuccessorCandidateCoreV3Error, match="frozen canonical identity"):
        verify_candidate_core_v3(
            repo_root=ROOT,
            core_path=CORE,
            expected_core_sha256="0" * 64,
        )
    shadow = tmp_path / CORE.name
    shadow.write_bytes(CORE.read_bytes())
    with pytest.raises(ChLtSuccessorCandidateCoreV3Error, match="path is not canonical"):
        verify_candidate_core_v3(
            repo_root=ROOT,
            core_path=shadow,
            expected_core_sha256=CORE_SHA256,
        )


def test_local_cli_emits_non_authoritative_v3_assessment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = audit_script.main(
        ["--core", str(CORE), "--expected-core-sha256", CORE_SHA256]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert result["status"] == STATUS
    assert result["candidate_core_frozen"] is False
    assert result["execution_authorized"] is False


def test_packaged_cli_is_sealed_and_reports_structured_v3_failures(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", lambda: None)
    assert packaged_cli.main(
        [
            "--evidence-root",
            str(ROOT),
            "--core",
            str(CORE),
            "--expected-core-sha256",
            "0" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert captured.out == ""
    assert failure["status"] == "INVALID_CANDIDATE_CORE_NO_GO"
    assert failure["candidate_core_admitted"] is False


def test_packaged_cli_rejects_unsealed_checkout_before_parsing_v3(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject() -> None:
        raise ReleaseCliIdentityError("checkout runtime is not sealed")

    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", _reject)
    assert packaged_cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "INVALID_GOVERNED_LT_RUNTIME"
