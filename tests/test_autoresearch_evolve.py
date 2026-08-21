"""test_autoresearch_evolve.py
---------------------------------

Regression tests for ``AutoResearchLoop.evolve()`` history accuracy.

Covers CR-02 (Phase 5 code review, 05-REVIEW.md): prior implementation
mutated ``current_rmse`` BEFORE computing the history record, then
re-derived the "action" field via ``trial.rmse < current_rmse + 0.001``.
This made nearly every iteration look like "keep" in the persisted
history (because after the mutation, ``trial.rmse == current_rmse``,
so the comparison was trivially true), and corrupted the recorded
``rmse_before`` value in the revert branch.

These tests confirm that:

  * ``action == "keep"`` only when ``trial.rmse < current_rmse`` strictly,
  * ``action == "revert"`` whenever the trial did NOT improve on the
    baseline (even for tiny RMSE deltas of order 1e-4),
  * ``rmse_before`` always reflects the PRE-decision baseline RMSE,
  * the ``improvements`` / ``reverts`` counters in the returned
    ``EvolutionResult`` match the count of each action in history.

The strategy: monkey-patch ``_run_backtest`` to return scripted
``BacktestResult`` objects so the test does not depend on PFC build,
fixtures, or numerics. Each test injects a known sequence of trial
RMSEs and asserts on the resulting history.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.pipeline import autoresearch as ar


def _make_dummy_epex() -> pd.DataFrame:
    """Return a minimal epex DataFrame; content is irrelevant because
    ``_run_backtest`` is monkey-patched. The shape just needs to satisfy
    any guard at the top of ``evolve()``.
    """
    idx = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
    return pd.DataFrame({"price_eur_mwh": 50.0}, index=idx)


def _scripted_backtest(rmses: list[float]):
    """Return a callable that produces ``BacktestResult`` objects with the
    scripted RMSE sequence on each successive call.

    The first call corresponds to the initial baseline; subsequent calls
    correspond to each trial iteration. Each result carries a stable
    per-agent RMSE map so the agent-selection logic in ``evolve()`` does
    not crash.
    """
    state = {"i": 0}
    per_agent = {"seasonal": 5.0, "weekday": 4.0, "hourly": 6.0,
                 "intraday": 3.0, "level": 2.0}

    def _call(*args, **kwargs):
        i = state["i"]
        state["i"] += 1
        rmse = rmses[min(i, len(rmses) - 1)]
        return ar.BacktestResult(
            rmse=rmse,
            mae=rmse * 0.8,
            bias=0.0,
            n_points=1000,
            per_agent_rmse=dict(per_agent),
            details=f"scripted call #{i}",
        )

    return _call


def test_evolve_history_keep_action_only_when_strictly_better(monkeypatch):
    """Iteration #1: trial improves (10.0 -> 9.5) → action=keep.

    Iteration #2: trial does NOT improve (9.5 -> 9.7) → action=revert.

    Regression for CR-02: previously iteration #2 was logged as 'keep'
    because the history record evaluated ``trial.rmse < current_rmse +
    0.001`` where current_rmse had already been overwritten with
    trial.rmse in the keep branch — so the check was tautological.
    """
    rmse_sequence = [10.0, 9.5, 9.7]
    monkeypatch.setattr(ar, "_run_backtest", _scripted_backtest(rmse_sequence))

    loop = ar.AutoResearchLoop(seed=42)
    result = loop.evolve(_make_dummy_epex(), n_iterations=2, test_months=3)

    assert len(loop.history) == 2, "Expected 2 history entries for 2 iterations"

    iter1, iter2 = loop.history

    assert iter1["action"] == "keep", (
        f"Iter 1 (10.0 -> 9.5) should be 'keep', got {iter1['action']!r}"
    )
    assert iter1["rmse_before"] == pytest.approx(10.0), (
        f"Iter 1 rmse_before should be 10.0 (baseline before keep), got "
        f"{iter1['rmse_before']!r}"
    )
    assert iter1["rmse_after"] == pytest.approx(9.5)

    assert iter2["action"] == "revert", (
        f"Iter 2 (9.5 -> 9.7) should be 'revert', got {iter2['action']!r}. "
        "CR-02 regression: history was incorrectly marking near-equal "
        "rmse iterations as 'keep' because the check was performed AFTER "
        "current_rmse had been mutated."
    )
    assert iter2["rmse_before"] == pytest.approx(9.5), (
        f"Iter 2 rmse_before should be 9.5 (current best, unchanged on "
        f"revert), got {iter2['rmse_before']!r}"
    )
    assert iter2["rmse_after"] == pytest.approx(9.7)

    # EvolutionResult counters must match history actions
    assert result.improvements == 1
    assert result.reverts == 1


def test_evolve_history_small_delta_revert(monkeypatch):
    """Trial RMSE only 0.0005 worse than baseline → must still be 'revert'.

    Regression for CR-02: the buggy check ``trial.rmse < current_rmse +
    0.001`` returned True for any trial within 1e-3 of the baseline, so
    a non-improvement of 5e-4 would have been logged as 'keep'. The
    correct semantic is the same as the keep/revert branch: strict
    ``trial.rmse < current_rmse``.
    """
    rmse_sequence = [10.0, 10.0005]
    monkeypatch.setattr(ar, "_run_backtest", _scripted_backtest(rmse_sequence))

    loop = ar.AutoResearchLoop(seed=42)
    result = loop.evolve(_make_dummy_epex(), n_iterations=1, test_months=3)

    assert len(loop.history) == 1
    entry = loop.history[0]
    assert entry["action"] == "revert", (
        f"Trial RMSE 10.0005 is NOT < baseline 10.0, action must be "
        f"'revert', got {entry['action']!r}. CR-02 regression."
    )
    assert entry["rmse_before"] == pytest.approx(10.0)
    assert entry["rmse_after"] == pytest.approx(10.0005)
    assert result.reverts == 1
    assert result.improvements == 0


def test_evolve_history_multiple_keeps_chain_rmse_before(monkeypatch):
    """Three successive improvements — each iteration's rmse_before must
    equal the previous iteration's rmse_after (the running best baseline).

    This catches a related CR-02 sub-bug: if ``rmse_before`` were taken
    AFTER the mutation, iteration N would report ``rmse_before ==
    rmse_after``, masking the actual delta.
    """
    rmse_sequence = [12.0, 11.0, 10.0, 9.0]
    monkeypatch.setattr(ar, "_run_backtest", _scripted_backtest(rmse_sequence))

    loop = ar.AutoResearchLoop(seed=42)
    result = loop.evolve(_make_dummy_epex(), n_iterations=3, test_months=3)

    assert len(loop.history) == 3
    expected_before = [12.0, 11.0, 10.0]
    expected_after = [11.0, 10.0, 9.0]
    for i, (entry, before, after) in enumerate(
        zip(loop.history, expected_before, expected_after)
    ):
        assert entry["action"] == "keep", (
            f"Iter {i + 1}: expected 'keep', got {entry['action']!r}"
        )
        assert entry["rmse_before"] == pytest.approx(before), (
            f"Iter {i + 1}: rmse_before should be {before} (running "
            f"baseline before keep), got {entry['rmse_before']!r}"
        )
        assert entry["rmse_after"] == pytest.approx(after)
    assert result.improvements == 3
    assert result.reverts == 0
    assert result.initial_rmse == pytest.approx(12.0)
    assert result.final_rmse == pytest.approx(9.0)
