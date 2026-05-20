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

import pytest


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
