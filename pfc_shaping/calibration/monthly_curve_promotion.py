"""Promotion checks for the governed monthly forward curve solver.

This module turns Phase F audit evidence into an explicit promotion decision.
It does not build curves and never mutates solver output.  The rules are
deliberately fail-closed: hard numerical gates must pass, critical rows always
block, and unsupported rows are accepted only for far-horizon shape evidence
when the threshold artifact proves an attempted point-in-time calibration with
insufficient sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


HARD_REQUIRED_GATES = frozenset(
    {
        "hard_monthly_curve_repricing",
        "neighbor_level_leakage",
    }
)

FAR_HORIZON_UNSUPPORTED_GATES = frozenset(
    {
        "same_month_rank_consistency",
        "residual_vs_implied_comparable_block",
    }
)

AGGREGATE_SHAPE_GATE = "monthly_shape_regression_2028_2030"


@dataclass(frozen=True)
class MonthlyCurvePromotionDecision:
    status: str
    approved: bool
    summary: dict[str, object]
    details: pd.DataFrame


def evaluate_monthly_curve_promotion(
    audit_gates: pd.DataFrame,
    historical_thresholds: pd.DataFrame,
    *,
    run_timestamp: pd.Timestamp | None = None,
    far_horizon_min_years: int = 2,
    required_hard_gates: set[str] | None = None,
    required_governance_gates: set[str] | None = None,
    manifest: Mapping[str, object] | None = None,
) -> MonthlyCurvePromotionDecision:
    """Evaluate whether monthly curve audit evidence is promotion-ready.

    ``far_horizon_min_years=2`` means delivery years at least run year + 2 may
    carry accepted unsupported shape evidence when the historical threshold
    artifact documents insufficient samples.  Near-horizon unsupported rows
    remain blocking.
    """

    gates = audit_gates.copy()
    thresholds = historical_thresholds.copy()
    required = set(required_hard_gates or HARD_REQUIRED_GATES)
    details: list[dict[str, object]] = []

    if gates.empty:
        details.append(_detail("audit_gates_present", "BLOCK", "audit_gates.csv is empty"))
        return _decision(details, gates, thresholds, manifest=manifest)
    missing_columns = sorted({"gate_id", "status"} - set(gates.columns))
    if missing_columns:
        details.append(
            _detail(
                "audit_gates_schema",
                "BLOCK",
                f"audit_gates.csv is missing required columns: {missing_columns}",
            )
        )
        return _decision(details, gates, thresholds, manifest=manifest)

    for gate_id in sorted(required):
        rows = gates[gates["gate_id"].astype(str).eq(gate_id)]
        if rows.empty:
            details.append(_detail(gate_id, "BLOCK", "required hard gate is missing"))
            continue
        bad = rows[~rows["status"].astype(str).eq("PASS")]
        if bad.empty:
            details.append(_detail(gate_id, "PASS", "required hard gate passed"))
        else:
            details.append(
                _detail(
                    gate_id,
                    "BLOCK",
                    f"required hard gate has non-PASS statuses: {sorted(bad['status'].astype(str).unique())}",
                )
            )

    for gate_id in sorted(required_governance_gates or set()):
        rows = gates[gates["gate_id"].astype(str).eq(gate_id)]
        if rows.empty:
            details.append(_detail(gate_id, "BLOCK", "required governance gate is missing"))
            continue
        bad = rows[~rows["status"].astype(str).eq("PASS")]
        if bad.empty:
            details.append(_detail(gate_id, "PASS", "required governance gate passed"))
        else:
            details.append(
                _detail(
                    gate_id,
                    "BLOCK",
                    f"required governance gate has non-PASS statuses: {sorted(bad['status'].astype(str).unique())}",
                )
            )

    critical = gates[gates["status"].astype(str).eq("CRITICAL")]
    if not critical.empty:
        for row in critical.itertuples(index=False):
            details.append(
                _detail(
                    str(getattr(row, "gate_id", "")),
                    "BLOCK",
                    "CRITICAL audit gate present",
                    year=_row_value(row, "year"),
                    month=_row_value(row, "month"),
                    product=_row_value(row, "product"),
                )
            )

    unsupported = gates[gates["status"].astype(str).eq("UNSUPPORTED")]
    unsupported_decisions: list[str] = []
    for idx, row in unsupported.iterrows():
        gate_id = str(row["gate_id"])
        if gate_id == AGGREGATE_SHAPE_GATE:
            continue
        accepted, reason = _unsupported_shape_row_is_accepted(
            row,
            thresholds,
            run_timestamp=run_timestamp,
            far_horizon_min_years=int(far_horizon_min_years),
        )
        unsupported_decisions.append("ACCEPT" if accepted else "BLOCK")
        details.append(
            _detail(
                gate_id,
                "ACCEPT" if accepted else "BLOCK",
                reason,
                year=row.get("year"),
                month=row.get("month"),
                product=row.get("product"),
                metric=row.get("metric_name"),
            )
        )

    aggregate = gates[gates["gate_id"].astype(str).eq(AGGREGATE_SHAPE_GATE)]
    if not aggregate.empty:
        agg_statuses = set(aggregate["status"].astype(str))
        if "CRITICAL" in agg_statuses:
            details.append(_detail(AGGREGATE_SHAPE_GATE, "BLOCK", "aggregate shape gate is CRITICAL"))
        elif "UNSUPPORTED" in agg_statuses:
            if unsupported_decisions and all(decision == "ACCEPT" for decision in unsupported_decisions):
                details.append(
                    _detail(
                        AGGREGATE_SHAPE_GATE,
                        "ACCEPT",
                        "aggregate unsupported accepted because all child unsupported rows are accepted far-horizon risks",
                    )
                )
            else:
                details.append(
                    _detail(
                        AGGREGATE_SHAPE_GATE,
                        "BLOCK",
                        "aggregate unsupported lacks accepted child far-horizon evidence",
                    )
                )
        elif agg_statuses == {"PASS"}:
            details.append(_detail(AGGREGATE_SHAPE_GATE, "PASS", "aggregate shape gate passed"))

    return _decision(details, gates, thresholds, manifest=manifest)


def _unsupported_shape_row_is_accepted(
    row: pd.Series,
    thresholds: pd.DataFrame,
    *,
    run_timestamp: pd.Timestamp | None,
    far_horizon_min_years: int,
) -> tuple[bool, str]:
    gate_id = str(row.get("gate_id", ""))
    if gate_id not in FAR_HORIZON_UNSUPPORTED_GATES:
        return False, "UNSUPPORTED is allowed only for calibrated shape-evidence gates"
    year = _numeric_or_none(row.get("year"))
    run_year = _run_year(run_timestamp, thresholds)
    if year is None or run_year is None:
        return False, "cannot classify horizon for UNSUPPORTED row"
    if int(year) < int(run_year) + int(far_horizon_min_years):
        return False, "UNSUPPORTED row is near-horizon or calibrable population"
    threshold = _matching_threshold(row, thresholds)
    if threshold is None:
        return False, "missing threshold evidence for UNSUPPORTED row"
    if str(threshold.get("status", "")).upper() != "UNSUPPORTED":
        return False, "threshold row is not UNSUPPORTED"
    n_snapshots = _numeric_or_none(threshold.get("n_snapshots"))
    min_required = _numeric_or_none(threshold.get("min_required_n"))
    if n_snapshots is None or min_required is None or n_snapshots >= min_required:
        return False, "threshold row does not prove insufficient sample"
    p90 = threshold.get("p90")
    p975 = threshold.get("p975")
    if pd.notna(p90) or pd.notna(p975):
        return False, "unsupported threshold row should not contain P90/P97.5"
    return True, "accepted far-horizon UNSUPPORTED with attempted threshold calibration and insufficient sample"


def _matching_threshold(row: pd.Series, thresholds: pd.DataFrame) -> pd.Series | None:
    if thresholds.empty:
        return None
    gate_id = str(row.get("gate_id", ""))
    metric = str(row.get("metric_name", ""))
    month = _numeric_or_none(row.get("month"))
    candidates = thresholds[
        thresholds["gate_id"].astype(str).eq(gate_id)
        & thresholds["metric"].astype(str).eq(metric)
    ].copy()
    if candidates.empty:
        return None
    buckets: list[str] = []
    if month is not None:
        buckets.append(f"month_{int(month):02d}")
    buckets.append("all")
    for bucket in buckets:
        match = candidates[candidates["delivery_bucket"].astype(str).eq(bucket)]
        if not match.empty:
            return match.iloc[0]
    return None


def _run_year(run_timestamp: pd.Timestamp | None, thresholds: pd.DataFrame) -> int | None:
    if run_timestamp is not None:
        return int(pd.Timestamp(run_timestamp).year)
    if thresholds.empty or "lookback_end" not in thresholds.columns:
        return None
    dates = pd.to_datetime(thresholds["lookback_end"], errors="coerce").dropna()
    if dates.empty:
        return None
    return int(dates.max().year)


def _decision(
    details: list[dict[str, object]],
    gates: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    manifest: Mapping[str, object] | None,
) -> MonthlyCurvePromotionDecision:
    frame = pd.DataFrame(details)
    blocking = int(frame["decision"].astype(str).eq("BLOCK").sum()) if not frame.empty else 0
    approved = blocking == 0
    status = "PROMOTION_EVIDENCE_PASS" if approved else "BLOCKED"
    summary = {
        "status": status,
        "approved": approved,
        "blocking_count": blocking,
        "audit_gate_status_counts": gates["status"].astype(str).value_counts().to_dict()
        if "status" in gates.columns
        else {},
        "threshold_status_counts": thresholds["status"].astype(str).value_counts().to_dict()
        if "status" in thresholds.columns
        else {},
        "manifest_solver_status": dict(manifest or {}).get("solver_status", ""),
        "manifest_gate_summary": dict(manifest or {}).get("gate_summary", {}),
    }
    return MonthlyCurvePromotionDecision(status=status, approved=approved, summary=summary, details=frame)


def _detail(
    gate_id: str,
    decision: str,
    reason: str,
    *,
    year: object = np.nan,
    month: object = np.nan,
    product: object = "",
    metric: object = "",
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "decision": decision,
        "reason": reason,
        "year": year,
        "month": month,
        "product": product,
        "metric": metric,
    }


def _numeric_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _row_value(row: object, name: str) -> object:
    return getattr(row, name, "")
