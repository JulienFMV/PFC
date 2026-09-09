from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_dependence_power_design as packaged_cli
import pfc_shaping.validation.ch_lt_dependence_power_design as power_design
import pfc_shaping.validation.ch_lt_successor_candidate_core_v2 as core_v2
import pfc_shaping.validation.ch_lt_successor_candidate_core_v3 as core_v3
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_dependence_power_design import (
    DESIGN_ID,
    DESIGN_RELATIVE_PATH,
    DESIGN_SHA256,
    REQUIRED_INPUTS,
    STATUS,
    ChLtDependencePowerDesignError,
    assess_dependence_power_design,
    compute_design_id,
    verify_dependence_power_design,
)
from scripts import audit_ch_lt_dependence_power_design as audit_script

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT.joinpath(*DESIGN_RELATIVE_PATH.split("/"))


def _document() -> dict[str, object]:
    return json.loads(DESIGN.read_text(encoding="utf-8"))


def _mutated_bytes(
    document: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    mutated_id = compute_design_id(document)
    document["design_id"] = mutated_id
    monkeypatch.setattr(power_design, "DESIGN_ID", mutated_id)
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    monkeypatch.setattr(power_design, "DESIGN_SHA256", hashlib.sha256(payload).hexdigest())
    return payload


def test_canonical_power_design_is_exact_outcome_blind_and_incomplete() -> None:
    result = verify_dependence_power_design(
        repo_root=ROOT,
        design_path=DESIGN,
        expected_design_sha256=DESIGN_SHA256,
    )

    assert hashlib.sha256(DESIGN.read_bytes()).hexdigest() == DESIGN_SHA256
    assert compute_design_id(_document()) == DESIGN_ID
    assert result["status"] == STATUS
    assert result["power_design_authored"] is True
    assert result["power_design_hash_closed_locally"] is True
    assert result["power_design_externally_frozen"] is False
    assert result["power_design_complete"] is False
    assert result["evidence_slot_satisfied"] is False
    assert result["missing_inputs"] == list(REQUIRED_INPUTS)
    assert [
        item["mechanical_expected_mean_block_lower_bound_origins"]
        for item in result["contrast_overlap_geometry"]
    ] == [1, 6, 6, 12, 12, 36]
    assert result["contrast_overlap_geometry"][-1]["primary_confirmatory_eligible"] is False
    calibration = _document()["power_calibration"]
    assert calibration["candidate_origin_count_grid"].startswith("COMMON_ASCENDING")
    assert calibration["contrast_specific_block_lengths_applied_on_common_grid"] is True
    assert "PARTIAL_NULL_CONFIGURATIONS" in calibration[
        "strong_fwer_configuration_scope"
    ]
    assert calibration["alternative_effects_distinct_from_null_margins"] is True
    assert "COMPLETE_FMV_GATEKEEPING_DECISION" in calibration[
        "conjunctive_power_acceptance"
    ]
    assert "SMALLEST_COMMON_N" in calibration["required_n_selection"]
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
    ):
        assert result[field] is False


def test_power_design_chain_reads_only_outcome_blind_t057_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readers = {
        power_design: power_design.read_stable_single_link_file,
        core_v3: core_v3.read_stable_single_link_file,
        core_v2: core_v2.read_stable_single_link_file,
    }
    observed: list[str] = []

    def _reader_for(module: object):
        original = readers[module]

        def _reader(path: Path, **kwargs: object) -> bytes:
            observed.append(path.as_posix())
            return original(path, **kwargs)

        return _reader

    for module in readers:
        monkeypatch.setattr(module, "read_stable_single_link_file", _reader_for(module))

    verify_dependence_power_design(
        repo_root=ROOT,
        design_path=DESIGN,
        expected_design_sha256=DESIGN_SHA256,
    )

    assert len(observed) == 12
    t057_reads = [path for path in observed if "T057" in path.upper()]
    assert len(t057_reads) == 1
    assert t057_reads[0].endswith("T057-OUTCOME-BLIND-TOMBSTONE-20260730.json")
    assert all("T057-EVIDENCE-SUPERSESSION-REGISTRY.json" not in path for path in observed)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("lifecycle", "externally_frozen", True),
        ("outcome_blind_firewall", "outcome_bearing_t057_registry", "ALLOWED"),
        ("contrast_geometry", "one_global_36_origin_rule_for_all_contrasts", 36),
        ("dependence_estimation", "independent_resampling_within_strata", "ALLOWED"),
        (
            "power_calibration",
            "post_selection_plugin_or_delete_one_required_n_as_bound",
            "ALLOWED",
        ),
        (
            "power_calibration",
            "strong_fwer_configuration_scope",
            "COMPLETE_NULL_ONLY",
        ),
    ],
)
def test_rehashed_policy_mutations_are_rejected(
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

    with pytest.raises(ChLtDependencePowerDesignError, match="semantic section mismatch"):
        assess_dependence_power_design(document, document_bytes=payload)


def test_rehashed_authority_and_missing_input_claims_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    document["design_complete"] = True
    inputs = document["required_inputs"]
    assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    inputs[0].update({"status": "SATISFIED", "path": "forged.json", "sha256": "a" * 64})
    payload = _mutated_bytes(document, monkeypatch)

    with pytest.raises(ChLtDependencePowerDesignError):
        assess_dependence_power_design(document, document_bytes=payload)


def test_wrong_hash_and_noncanonical_path_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ChLtDependencePowerDesignError, match="canonical identity"):
        verify_dependence_power_design(
            repo_root=ROOT,
            design_path=DESIGN,
            expected_design_sha256="0" * 64,
        )
    shadow = tmp_path / DESIGN.name
    shadow.write_bytes(DESIGN.read_bytes())
    with pytest.raises(ChLtDependencePowerDesignError, match="path is not canonical"):
        verify_dependence_power_design(
            repo_root=ROOT,
            design_path=shadow,
            expected_design_sha256=DESIGN_SHA256,
        )


def test_local_cli_reports_valid_incomplete_design(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert audit_script.main(
        ["--design", str(DESIGN), "--expected-design-sha256", DESIGN_SHA256]
    ) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    assert result["status"] == STATUS
    assert result["power_design_complete"] is False
    assert result["execution_authorized"] is False


def test_packaged_cli_is_sealed_and_failures_are_structured(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", lambda: None)
    assert packaged_cli.main(
        [
            "--evidence-root",
            str(ROOT),
            "--design",
            str(DESIGN),
            "--expected-design-sha256",
            "0" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.err)
    assert captured.out == ""
    assert result["status"] == "INVALID_DEPENDENCE_POWER_DESIGN_NO_GO"
    assert result["evidence_slot_satisfied"] is False


def test_packaged_cli_rejects_unsealed_checkout_before_parsing(
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
