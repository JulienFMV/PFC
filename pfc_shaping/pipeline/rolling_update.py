"""Tombstone for the retired direct PFC publisher.

All LT publication must use the governed candidate, audit, receipt and atomic
promotion workflow. The historical implementation was intentionally removed:
it mixed ingestion, model execution, OMPEX benchmarking and publication in one
callable module.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["run_update", "setup_logging"]


def setup_logging(run_date: str) -> None:
    """Retained symbol for callers that still import the legacy module."""

    raise RuntimeError(
        "Legacy rolling_update logging is disabled with the retired publisher"
    )


def run_update(config: dict | None = None) -> Path:
    """Reject the retired direct publisher before any I/O or ingestion."""

    raise RuntimeError(
        "Legacy rolling_update publication is disabled; use the governed "
        "candidate/audit/receipt/promotion workflow"
    )


def _run_update_legacy(config: dict | None = None) -> Path:
    """Reject direct access to the removed historical implementation."""

    raise RuntimeError(
        "Retired rolling_update implementation cannot be invoked; use the "
        "governed candidate/audit/receipt/promotion workflow"
    )


if __name__ == "__main__":
    run_update()
