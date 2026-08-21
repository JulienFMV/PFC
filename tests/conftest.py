"""
conftest.py
-----------
Project-wide pytest fixtures.

The ``_pfc_lt_env_hygiene`` autouse fixture snapshots all ``PFC_LT_*``
env-var keys before each test and restores them after, preventing leak
between tests that set the ``PFC_LT_USE_SEASONAL_HOURLY_SHAPE`` flag (or
any future PFC_LT_* keys).

Per Phase 5bis-A decision D-12 (.planning/phases/05B-shape-hourly-
infrastructure-flag-no-op-refactor/05B-CONTEXT.md).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

import pfc_shaping.cli.governed_release as governed_release_cli
import pfc_shaping.pipeline.atomic_promotion as atomic_promotion
import pfc_shaping.pipeline.candidate_evidence as candidate_evidence
import pfc_shaping.pipeline.governed_release as governed_release_pipeline
import pfc_shaping.pipeline.governed_release_cli_contract as release_cli_contract

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_PYTEST_WORKSPACE_ROOT = _WORKSPACE_ROOT / "build" / "test-workspaces" / "pytest"


class _GovernedTempPathFactory:
    """Minimal pytest temp factory constrained to the canonical workspace."""

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        if (
            not isinstance(basename, str)
            or not basename
            or any(character in basename for character in ("/", "\\", ":"))
            or basename in {".", ".."}
        ):
            raise ValueError("pytest temporary basename is unsafe")
        suffix = f"-{uuid.uuid4().hex}" if numbered else ""
        selected = _PYTEST_WORKSPACE_ROOT / f"{basename}{suffix}"
        selected.mkdir(parents=True, exist_ok=False)
        return selected


@pytest.fixture(scope="session")
def tmp_path_factory() -> _GovernedTempPathFactory:
    """Keep all mutable pytest directories below the canonical workspace."""

    return _GovernedTempPathFactory()


@pytest.fixture
def tmp_path(tmp_path_factory: _GovernedTempPathFactory) -> Path:
    """Workspace-local replacement for pytest's host-temp fixture."""

    return tmp_path_factory.mktemp("test")


@pytest.fixture(autouse=True)
def _pfc_lt_env_hygiene():
    """Snapshot all PFC_LT_* env-var keys before each test and restore after.

    Prevents test-to-test env-var leakage for any key matching ``PFC_LT_*``
    (in particular ``PFC_LT_USE_SEASONAL_HOURLY_SHAPE`` used by ShapeHourly).

    Known PFC_LT_* keys covered by the prefix snapshot (no code change needed
    when new keys are added — the prefix match handles them automatically):
    - PFC_LT_USE_SEASONAL_HOURLY_SHAPE (5bis-A D-06, ShapeHourly freeze-at-init)
    - PFC_LT_ALLOW_NEGATIVE_PRICES (Phase 5 D-A2-2, PFCAssembler — introduced in Plan 05-03)
    """
    # Snapshot all PFC_LT_* keys present at test entry
    snapshot = {k: v for k, v in os.environ.items() if k.startswith("PFC_LT_")}
    # Yield to test — test may freely set/del env vars
    yield
    # Restore: delete keys that were absent, restore keys that were present
    current = {k for k in os.environ if k.startswith("PFC_LT_")}
    for k in current - set(snapshot):
        os.environ.pop(k, None)
    for k, v in snapshot.items():
        os.environ[k] = v


@pytest.fixture(autouse=True)
def _sealed_runtime_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise governance as an installed wheel rather than an unsealed checkout."""

    candidate_revision = "1" * 40
    runtime_revision = "1" * 64
    monkeypatch.setattr(candidate_evidence, "SOURCE_REVISION", candidate_revision)
    monkeypatch.setattr(release_cli_contract, "SOURCE_REVISION", runtime_revision)
    monkeypatch.setattr(
        release_cli_contract,
        "_installed_runtime_source_revision",
        lambda: runtime_revision,
    )
    monkeypatch.setattr(
        release_cli_contract,
        "_assert_isolated_runtime_flags",
        lambda: None,
    )
    monkeypatch.setattr(
        release_cli_contract,
        "_assert_launcherless_runtime_admission",
        lambda: None,
    )
    monkeypatch.setattr(
        release_cli_contract,
        "_installed_runtime_dependency_versions",
        lambda: dict(release_cli_contract.REQUIRED_RUNTIME_DISTRIBUTIONS),
    )
    monkeypatch.setattr(
        release_cli_contract,
        "assert_production_transition_runtime_authorized",
        lambda: None,
    )
    monkeypatch.setattr(
        atomic_promotion,
        "assert_production_transition_runtime_authorized",
        lambda: None,
    )
    monkeypatch.setattr(
        governed_release_pipeline,
        "assert_production_transition_runtime_authorized",
        lambda: None,
    )
    monkeypatch.setattr(
        governed_release_cli,
        "assert_production_transition_runtime_authorized",
        lambda: None,
    )
