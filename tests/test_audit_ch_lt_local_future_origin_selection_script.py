from __future__ import annotations

from pathlib import Path

import pytest

from pfc_shaping.validation import ch_lt_local_future_origin_selection as auditor


def test_audits_recursive_outcome_blind_future_origin_selection() -> None:
    result = auditor.audit_selection(
        registry_path=auditor.ROOT / auditor.REGISTRY_RELATIVE,
        expected_registry_sha256=auditor.REGISTRY_SHA256,
    )

    assert result["status"] == (
        "VALID_LOCAL_FUTURE_ORIGIN_REHEARSAL_V7_PACKAGED_MODULE_"
        "SOURCES_BOUND_NONCOUNTABLE_NO_GO"
    )
    assert result["target_count"] == 36
    assert result["prediction_count"] == 108
    assert result["recursive_binding_count"] == 25
    assert result["origin_precedes_first_target"] is True
    assert result["outcome_blind"] is True
    assert result["countable_prospective_origin"] is False
    assert result["future_holdout_consumed"] is False
    assert result["t057_consumed"] is False
    assert result["scientific_admission"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False


def test_rejects_registry_hash_mismatch() -> None:
    with pytest.raises(
        auditor.LocalFutureOriginSelectionError,
        match="does not match frozen identity",
    ):
        auditor.audit_selection(
            registry_path=auditor.ROOT / auditor.REGISTRY_RELATIVE,
            expected_registry_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "relative",
    ["../escape.json", "build\\escape.json", "C:/escape.json", "build/a:b.json"],
)
def test_rejects_unsafe_linked_paths(relative: str) -> None:
    with pytest.raises(auditor.LocalFutureOriginSelectionError, match="path is invalid"):
        auditor._safe_path(auditor.ROOT, relative, label="fixture")


def test_rejects_any_authority_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    original = auditor._read_json

    def _tampered(path: Path, expected_sha256: str, *, label: str):
        document = original(path, expected_sha256, label=label)
        if label != "selection":
            return document
        changed = dict(document)
        boundary = dict(changed["authority_boundary"])
        boundary["countable_prospective_origin"] = True
        changed["authority_boundary"] = boundary
        unsigned = dict(changed)
        unsigned.pop("selection_id", None)
        changed["selection_id"] = auditor.hashlib.sha256(
            auditor._canonical(unsigned)
        ).hexdigest()
        return changed

    monkeypatch.setattr(auditor, "_read_json", _tampered)
    with pytest.raises(
        auditor.LocalFutureOriginSelectionError,
        match="authority boundary is invalid",
    ):
        auditor.audit_selection(
            registry_path=auditor.ROOT / auditor.REGISTRY_RELATIVE,
            expected_registry_sha256=auditor.REGISTRY_SHA256,
        )


def test_rejects_unknown_top_level_field(monkeypatch: pytest.MonkeyPatch) -> None:
    original = auditor._read_json

    def _tampered(path: Path, expected_sha256: str, *, label: str):
        document = original(path, expected_sha256, label=label)
        if label != "selection":
            return document
        changed = dict(document)
        changed["outcome_payload"] = {"forbidden": True}
        return changed

    monkeypatch.setattr(auditor, "_read_json", _tampered)
    with pytest.raises(
        auditor.LocalFutureOriginSelectionError,
        match="keys are not exact",
    ):
        auditor.audit_selection(
            registry_path=auditor.ROOT / auditor.REGISTRY_RELATIVE,
            expected_registry_sha256=auditor.REGISTRY_SHA256,
        )


def test_rejects_scoring_plan_change(monkeypatch: pytest.MonkeyPatch) -> None:
    original = auditor._read_json

    def _tampered(path: Path, expected_sha256: str, *, label: str):
        document = original(path, expected_sha256, label=label)
        if label != "selection":
            return document
        changed = dict(document)
        plan = dict(changed["evaluation_contract"])
        plan["scenario_pooling"] = "ALLOWED"
        changed["evaluation_contract"] = plan
        unsigned = dict(changed)
        unsigned.pop("selection_id", None)
        changed["selection_id"] = auditor.hashlib.sha256(
            auditor._canonical(unsigned)
        ).hexdigest()
        return changed

    monkeypatch.setattr(auditor, "_read_json", _tampered)
    with pytest.raises(
        auditor.LocalFutureOriginSelectionError,
        match="evaluation contract is invalid",
    ):
        auditor.audit_selection(
            registry_path=auditor.ROOT / auditor.REGISTRY_RELATIVE,
            expected_registry_sha256=auditor.REGISTRY_SHA256,
        )
