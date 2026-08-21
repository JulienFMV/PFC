from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pfc_shaping.cli import audit_ch_market_time_regime as cli_module
from pfc_shaping.package_contract import ALLOWED_RUNTIME_PYTHON_FILES
from pfc_shaping.validation.ch_market_time_regime import (
    CONTRACT_ID,
    DOCUMENT_SHA256,
    STATUS,
    ChMarketTimeRegimeError,
    assess_market_time_regime,
    compute_contract_id,
    load_market_time_regime_contract,
    verify_bound_evidence_files,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-MARKET-TIME-REGIME-CONTRACT-20260730.json"
)


def _document() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_canonical_contract_separates_model_grid_from_native_truth() -> None:
    payload = CONTRACT.read_bytes()
    document = json.loads(payload)

    assessment = assess_market_time_regime(document, document_bytes=payload)

    assert hashlib.sha256(payload).hexdigest() == DOCUMENT_SHA256
    assert assessment.contract_id == CONTRACT_ID == compute_contract_id(document)
    assert assessment.status == STATUS
    assert assessment.evidence_files_verified is False
    result = assessment.to_dict()
    assert result["native_day_ahead_market_time_unit_minutes"] == 60
    assert result["native_intraday_auction_market_time_unit_minutes"] == 60
    assert result["planned_first_trading_date"] == "2026-11-03"
    assert result["effective_first_delivery_start_utc"] is None
    assert result["transition_admitted"] is False
    assert result["production_authorization"] is False
    assert result["blockers"]


    assert len(result["blockers"]) == 8


def test_transition_auditor_is_in_sealed_wheel_without_transition_authority() -> None:
    assert "pfc_shaping/validation/ch_market_time_regime.py" in ALLOWED_RUNTIME_PYTHON_FILES
    assert "pfc_shaping/cli/audit_ch_market_time_regime.py" in ALLOWED_RUNTIME_PYTHON_FILES


def test_loader_binds_canonical_path_and_caller_held_hash() -> None:
    document, payload = load_market_time_regime_contract(
        CONTRACT, expected_sha256=DOCUMENT_SHA256
    )
    assert document == json.loads(payload)

    with pytest.raises(ChMarketTimeRegimeError, match="frozen canonical identity"):
        load_market_time_regime_contract(CONTRACT, expected_sha256="0" * 64)
    with pytest.raises(ChMarketTimeRegimeError, match="not canonical"):
        load_market_time_regime_contract(
            ROOT / "shadow-market-time-contract.json",
            expected_sha256=DOCUMENT_SHA256,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "claim_native_qh",
        "confirm_go_live",
        "invent_delivery_boundary",
        "admit_transition",
        "allow_proxy_training",
        "grant_production",
        "change_planned_date",
    ],
)
def test_rehashed_policy_weakening_cannot_replace_canonical_contract(
    mutation: str,
) -> None:
    document = copy.deepcopy(_document())
    regimes = document["regimes"]
    gate = document["transition_admission_gate"]
    policy = document["data_and_validation_policy"]
    controls = document["execution_controls"]
    assert isinstance(regimes, dict)
    assert isinstance(gate, dict)
    assert isinstance(policy, dict)
    assert isinstance(controls, dict)
    if mutation == "claim_native_qh":
        regimes["pre_transition"]["native_quarter_hour_truth_claim_possible"] = True
    elif mutation == "confirm_go_live":
        regimes["planned_transition"]["confirmed_go_live"] = True
    elif mutation == "invent_delivery_boundary":
        regimes["planned_transition"]["effective_first_delivery_start_utc"] = (
            "2026-11-03T23:00:00Z"
        )
    elif mutation == "admit_transition":
        gate["transition_admitted"] = True
    elif mutation == "allow_proxy_training":
        policy["proxy_rows_may_train_or_score_native_quarter_hour_layer"] = True
    elif mutation == "grant_production":
        controls["contract_authorizes_production"] = True
    else:
        regimes["planned_transition"]["latest_locally_observed_official_plan"][
            "planned_first_trading_date"
        ] = "2026-10-01"
    document["contract_id"] = compute_contract_id(document)
    mutated = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(ChMarketTimeRegimeError, match="frozen canonical identity"):
        assess_market_time_regime(document, document_bytes=mutated)


def test_mapping_must_match_captured_bytes() -> None:
    document = _document()
    controls = document["execution_controls"]
    assert isinstance(controls, dict)
    controls["promotion_gate"] = True

    with pytest.raises(ChMarketTimeRegimeError, match="differs from captured bytes"):
        assess_market_time_regime(document, document_bytes=CONTRACT.read_bytes())


def test_evidence_verification_requires_repository_root() -> None:
    with pytest.raises(ChMarketTimeRegimeError, match="repo_root is required"):
        assess_market_time_regime(
            _document(),
            document_bytes=CONTRACT.read_bytes(),
            verify_evidence_files=True,
        )


def test_evidence_verifier_rejects_missing_or_mismatched_bytes(tmp_path: Path) -> None:
    document = _document()
    first = document["evidence"]["official_source_captures"][0]
    target = tmp_path.joinpath(*Path(first["path"]).parts)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not-the-bound-epex-page")

    with pytest.raises(ChMarketTimeRegimeError, match="evidence size mismatch"):
        verify_bound_evidence_files(document, repo_root=tmp_path)


def test_evidence_verifier_rejects_repository_escape(tmp_path: Path) -> None:
    document = _document()
    document["evidence"]["official_source_captures"][0]["path"] = "../escape.bin"

    with pytest.raises(ChMarketTimeRegimeError, match="repository-relative"):
        verify_bound_evidence_files(document, repo_root=tmp_path)


def test_evidence_verifier_binds_superseded_document_bytes(tmp_path: Path) -> None:
    document = _document()
    bindings = [
        (item, "path", "size_bytes", "sha256")
        for item in document["evidence"]["official_source_captures"]
    ]
    hourly = document["evidence"]["local_hourly_observation"]
    for prefix in (
        "selection_manifest",
        "selected_ledger",
        "selected_execution_receipt",
    ):
        bindings.append(
            (
                hourly,
                f"{prefix}_path",
                f"{prefix}_size_bytes",
                f"{prefix}_sha256",
            )
        )
    bindings.append(
        (
            document["supersession"]["superseded_document"],
            "path",
            "size_bytes",
            "sha256",
        )
    )
    last_relative = ""
    for index, (binding, path_key, size_key, hash_key) in enumerate(bindings):
        payload = f"evidence-{index}".encode()
        relative = f"evidence/item-{index}.bin"
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        binding[path_key] = relative
        binding[size_key] = len(payload)
        binding[hash_key] = hashlib.sha256(payload).hexdigest()
        last_relative = relative

    verify_bound_evidence_files(document, repo_root=tmp_path)
    (tmp_path / last_relative).write_bytes(b"changed")
    with pytest.raises(ChMarketTimeRegimeError, match="evidence size mismatch"):
        verify_bound_evidence_files(document, repo_root=tmp_path)


def test_durable_audit_route_is_runtime_bound_and_exactly_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "AGENTS.md").write_text("test", encoding="utf-8")
    (root / "build" / "market-time-regime-audits").mkdir(parents=True)
    contract = root / "contract.json"
    contract.write_bytes(b"contract")
    runtime_receipt = root / "runtime-receipt.json"
    runtime_receipt.write_bytes(b"runtime")
    output = root / "build" / "market-time-regime-audits" / "audit.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        cli_module, "assert_installed_runtime_sealed", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        cli_module,
        "load_market_time_regime_contract",
        lambda *_args, **_kwargs: ({"contract": True}, b"contract"),
    )
    base = {
        "schema_version": "ch_market_time_regime_audit.v1",
        "contract_id": CONTRACT_ID,
        "document_sha256": DOCUMENT_SHA256,
        "status": STATUS,
        "evidence_files_verified": True,
        "transition_admitted": False,
        "scientific_admission": False,
        "execution_authorized": False,
        "production_authorization": False,
        "promotion_gate": False,
        "blockers": ["MISSING_GO_LIVE"],
    }
    monkeypatch.setattr(
        cli_module,
        "assess_market_time_regime",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: dict(base)),
    )
    monkeypatch.setattr(
        cli_module,
        "_runtime_identity",
        lambda **_kwargs: {
            "source_revision": "f" * 64,
            "runtime_receipt_sha256": "a" * 64,
            "sys_path": ["Lib", "DLLs", "governed-site-packages"],
        },
    )
    kwargs = {
        "contract_path": contract,
        "expected_contract_sha256": DOCUMENT_SHA256,
        "repo_root": root,
        "runtime_receipt_path": runtime_receipt,
        "expected_runtime_receipt_sha256": "a" * 64,
        "output_path": output,
    }
    result = cli_module.audit_market_time_regime(**kwargs)
    first = output.read_bytes()
    retry = cli_module.audit_market_time_regime(**kwargs)

    assert result == retry
    assert output.read_bytes() == first
    assert result["status"] == STATUS
    assert result["receipt_status"] == "LOCAL_DIAGNOSTIC_NOT_AUTHORITY"
    assert result["evidence_files_verified"] is True
    assert result["transition_admitted"] is False
    assert result["production_authorization"] is False
    assert result["runtime_identity"]["runtime_receipt_sha256"] == "a" * 64
    assert result["cwd_equals_repo_root"] is True
    assert result["normalized_command_sha256"]
    assert result["audit_operation_id"]


def test_cli_argument_errors_are_structured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_module.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    failure = json.loads(captured.err)
    assert failure["status"] == "INVALID_CH_MARKET_TIME_REGIME_ARGUMENTS"
    assert failure["production_authorization"] is False


def test_checkout_wrapper_resolves_repo_module_and_fails_structured() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "scripts/audit_ch_market_time_regime.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    failure = json.loads(result.stderr)
    assert failure["status"] == "INVALID_CH_MARKET_TIME_REGIME_ARGUMENTS"
    assert failure["production_authorization"] is False
