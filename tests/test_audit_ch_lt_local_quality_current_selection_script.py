from __future__ import annotations

import copy

import pytest

from scripts import audit_ch_lt_local_quality_current_selection as audit


def test_current_local_quality_selection_chain_is_exact() -> None:
    result = audit.verify_current_selection(audit.ROOT)

    assert result["status"] == audit.STATUS
    assert result["selection_id"] == audit.SELECTION_ID
    assert result["model_run_count"] == 2
    assert result["independent_rolling_origin_count"] == 0
    assert result["future_holdout_consumed"] is False
    assert result["t057_consumed"] is False
    assert result["model_selection_eligible"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False


def test_current_local_quality_selection_rejects_rebound_hash(monkeypatch) -> None:
    monkeypatch.setattr(audit, "SELECTION_SHA256", "0" * 64)

    with pytest.raises(audit.CurrentLocalQualitySelectionError, match="SHA-256 mismatch"):
        audit.verify_current_selection(audit.ROOT)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda payload: payload["deadline"].__setitem__("deadline_exceeded", True), "binding"),
        (
            lambda payload: payload["process_tree"].__setitem__(
                "kill_on_supervisor_exit", False
            ),
            "binding",
        ),
        (
            lambda payload: payload["worker_capability"].__setitem__("consumed", False),
            "binding",
        ),
    ],
)
def test_current_selection_rejects_incomplete_qualification_supervision(
    monkeypatch,
    mutation,
    expected,
) -> None:
    original = audit._read_bound

    def rebound(*args, **kwargs):
        value, payload = original(*args, **kwargs)
        if kwargs.get("label") == "laptop qualification supervisor receipt":
            value = copy.deepcopy(value)
            mutation(value)
        return value, payload

    monkeypatch.setattr(audit, "_read_bound", rebound)
    with pytest.raises(audit.CurrentLocalQualitySelectionError, match=expected):
        audit.verify_current_selection(audit.ROOT)
