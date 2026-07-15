"""Private config-neutral BASE input derivation for Tier 2 SELECTION.

This module is not an evidence boundary. It deliberately emits no target row,
price, candidate config, prediction, metric, rank, or campaign claim. A later
path-only signed package verifier must authenticate and replay these records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import pandas as pd

import pfc_shaping.calibration.tier2_monthly_eex_base_replay as base_replay_module
import pfc_shaping.calibration.tier2_monthly_eex_fold_evidence as fold_module
from pfc_shaping.calibration.monthly_forward_curve import product_periods
from pfc_shaping.calibration.tier2_monthly_eex_evaluation import (
    EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE,
    Tier2MonthlyEexEvaluationError,
    VerifiedEvaluationPlan,
    _sha256_json,
)


SELECTION_BASE_INPUT_SET_SCHEMA = "tier2_monthly_eex_selection_base_input_set.v1"
SELECTION_BASE_INPUT_RECORD_SCHEMA = (
    "tier2_monthly_eex_selection_base_input_record.v1"
)
MAX_SELECTION_FOLDS = int(
    EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE["max_selection_folds"]
)
MAX_VERIFIED_FRAME_ROWS = int(
    EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE["max_verified_frame_rows"]
)
MAX_FOLD_ROW_WORK = int(
    EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE["max_fold_row_work"]
)
MAX_DELIVERY_MONTHS_PER_FOLD = int(
    EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE["max_delivery_months_per_fold"]
)
MAX_TOTAL_EMITTED_ROW_HASHES = int(
    EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE["max_total_emitted_row_hashes"]
)
UNSUPPORTED_NON_BASE_SELECTION = (
    "UNSUPPORTED_REQUIRES_JOINT_PEAK_OFFPEAK_SOLVER"
)


@dataclass(frozen=True)
class _SelectionBaseInputRecord:
    schema_version: str
    fold_id: str
    fold_spec_hash: str
    origin_id: str
    fold_as_of_utc: str
    target_product: str
    target_load_type: str
    evaluation_snapshot_id: str
    snapshot_available_at_utc: str
    delivery_months: tuple[str, ...]
    base_history_row_hashes: tuple[str, ...]
    base_retained_row_hashes: tuple[str, ...]
    base_input_set_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        for field_name in (
            "delivery_months",
            "base_history_row_hashes",
            "base_retained_row_hashes",
        ):
            payload[field_name] = list(payload[field_name])
        return payload


@dataclass(frozen=True)
class _SelectionBaseInputSet:
    schema_version: str
    plan_id: str
    plan_artifact_sha256: str
    fold_count: int
    ordered_fold_ids: tuple[str, ...]
    records: tuple[_SelectionBaseInputRecord, ...]
    input_set_id: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "plan_artifact_sha256": self.plan_artifact_sha256,
            "fold_count": self.fold_count,
            "ordered_fold_ids": list(self.ordered_fold_ids),
            "records": [record.canonical_payload() for record in self.records],
            "input_set_id": self.input_set_id,
        }


def _derive_selection_base_input_set(
    frame: pd.DataFrame,
    *,
    plan: object,
) -> _SelectionBaseInputSet:
    """Derive every preregistered SELECTION input without choosing a config."""

    payload = _verified_plan_payload(plan)
    folds = _preflight_selection_folds(payload)
    if len(frame) > MAX_VERIFIED_FRAME_ROWS:
        raise Tier2MonthlyEexEvaluationError(
            "verified EEX frame exceeds governed row bound"
        )
    if len(frame) * len(folds) > MAX_FOLD_ROW_WORK:
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION fold-row work exceeds governed resource bound"
        )
    if len(frame) * len(folds) * 2 > MAX_TOTAL_EMITTED_ROW_HASHES:
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION emitted row hashes may exceed governed resource bound"
        )
    if frame["row_hash"].astype(str).duplicated().any():
        raise Tier2MonthlyEexEvaluationError(
            "verified EEX frame contains duplicate row hashes"
        )
    for fold in folds:
        _, _, _, delivery_months = _derive_solver_input_frames(
            frame,
            plan=plan,
            fold_spec=fold,
        )
        if len(delivery_months) > MAX_DELIVERY_MONTHS_PER_FOLD:
            raise Tier2MonthlyEexEvaluationError(
                "SELECTION delivery grid exceeds governed resource bound"
            )

    records = tuple(
        _derive_selection_base_input(frame, plan=plan, fold_spec=fold)
        for fold in folds
    )
    ordered_fold_ids = tuple(record.fold_id for record in records)
    if len(records) != len(folds) or len(set(ordered_fold_ids)) != len(records):
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION input fold inventory is incomplete or duplicated"
        )
    unsigned: dict[str, object] = {
        "schema_version": SELECTION_BASE_INPUT_SET_SCHEMA,
        "plan_id": str(plan.plan_id),
        "plan_artifact_sha256": str(plan.artifact_sha256),
        "fold_count": len(records),
        "ordered_fold_ids": list(ordered_fold_ids),
        "records": [record.canonical_payload() for record in records],
    }
    return _SelectionBaseInputSet(
        schema_version=SELECTION_BASE_INPUT_SET_SCHEMA,
        plan_id=str(plan.plan_id),
        plan_artifact_sha256=str(plan.artifact_sha256),
        fold_count=len(records),
        ordered_fold_ids=ordered_fold_ids,
        records=records,
        input_set_id=_sha256_json(unsigned),
    )


def _derive_selection_base_input(
    frame: pd.DataFrame,
    *,
    plan: object,
    fold_spec: Mapping[str, object],
) -> _SelectionBaseInputRecord:
    """Derive one solver-consumable BASE row inventory without scoring data."""

    _verified_plan_payload(plan)
    if str(fold_spec.get("campaign_type", "")) != "SELECTION":
        raise Tier2MonthlyEexEvaluationError("input fold is not SELECTION")
    if str(fold_spec.get("target_load_type", "")) != "BASE":
        raise Tier2MonthlyEexEvaluationError(UNSUPPORTED_NON_BASE_SELECTION)
    registered = fold_module._find_selection_fold(
        plan.payload, str(fold_spec.get("fold_id", ""))
    )
    if dict(registered) != dict(fold_spec):
        raise Tier2MonthlyEexEvaluationError(
            "input fold differs from exact preregistered specification"
        )

    historical_frame, retained_frame, target_identity, delivery_months = (
        _derive_solver_input_frames(
            frame,
            plan=plan,
            fold_spec=fold_spec,
        )
    )
    if len(delivery_months) > MAX_DELIVERY_MONTHS_PER_FOLD:
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION delivery grid exceeds governed resource bound"
        )
    historical = _ordered_base_hashes(
        historical_frame,
        label="SELECTION BASE historical context",
    )
    retained = _ordered_base_hashes(
        retained_frame,
        label="SELECTION BASE retained snapshot",
    )
    if set(historical).intersection(retained):
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION BASE historical and retained inputs overlap"
        )
    target_hash = str(target_identity["row_hash"])
    if target_hash in historical or target_hash in retained:
        raise Tier2MonthlyEexEvaluationError(
            "masked target remains in SELECTION BASE inputs"
        )
    input_hash = base_replay_module._base_input_set_sha256(
        historical,
        retained,
    )
    return _SelectionBaseInputRecord(
        schema_version=SELECTION_BASE_INPUT_RECORD_SCHEMA,
        fold_id=str(fold_spec["fold_id"]),
        fold_spec_hash=str(fold_spec["fold_spec_hash"]),
        origin_id=str(fold_spec["origin_id"]),
        fold_as_of_utc=str(fold_spec["fold_as_of_utc"]),
        target_product=str(fold_spec["target_product"]),
        target_load_type="BASE",
        evaluation_snapshot_id=str(target_identity["snapshot_id"]),
        snapshot_available_at_utc=pd.Timestamp(
            target_identity["available_at"]
        ).isoformat(),
        delivery_months=tuple(delivery_months),
        base_history_row_hashes=historical,
        base_retained_row_hashes=retained,
        base_input_set_sha256=input_hash,
    )


def _derive_solver_input_frames(
    frame: pd.DataFrame,
    *,
    plan: VerifiedEvaluationPlan,
    fold_spec: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object], tuple[str, ...]]:
    """Derive only model-worker inputs; never build diagnostic/scored roles."""

    origin = fold_module._aware_utc(
        fold_spec["fold_as_of_utc"], label="fold_as_of_utc"
    )
    ch = frame[frame["market"].astype(str).eq("CH")]
    eligible = ch[ch["available_at"] <= origin]
    if eligible.empty:
        raise Tier2MonthlyEexEvaluationError(
            "UNSUPPORTED_SNAPSHOT_OR_TARGET_UNAVAILABLE"
        )
    latest_available = eligible["available_at"].max()
    latest = eligible[eligible["available_at"].eq(latest_available)]
    snapshot_ids = sorted(set(latest["snapshot_id"].astype(str)))
    if len(snapshot_ids) != 1:
        raise Tier2MonthlyEexEvaluationError(
            "UNSUPPORTED_SNAPSHOT_OR_TARGET_UNAVAILABLE"
        )
    snapshot = ch[ch["snapshot_id"].astype(str).eq(snapshot_ids[0])]
    target_matches = snapshot[
        snapshot["product"].astype(str).eq(str(fold_spec["target_product"]))
        & snapshot["load_type"].astype(str).eq("BASE")
    ]
    if len(target_matches) != 1:
        raise Tier2MonthlyEexEvaluationError(
            "UNSUPPORTED_SNAPSHOT_OR_TARGET_UNAVAILABLE"
        )
    target_columns = (
        "product",
        "load_type",
        "row_hash",
        "snapshot_id",
        "source_document_id",
        "available_at",
    )
    target_identity = target_matches.loc[:, target_columns].iloc[0].to_dict()
    target_months = set(product_periods(str(target_identity["product"])))
    overlap = snapshot["product"].astype(str).map(
        lambda product: bool(target_months.intersection(product_periods(product)))
    )
    retained = snapshot[~overlap]
    historical = fold_module._context_rows(
        frame,
        cutoff=origin,
        evaluation_snapshot_id=snapshot_ids[0],
        evaluation_document_id=str(target_identity["source_document_id"]),
    )
    base_retained = retained[
        retained["market"].astype(str).eq("CH")
        & retained["load_type"].astype(str).eq("BASE")
    ]
    delivery_periods = sorted(
        {
            month
            for product in [
                str(target_identity["product"]),
                *base_retained["product"].astype(str).tolist(),
            ]
            for month in product_periods(product)
        }
    )
    if not delivery_periods:
        raise Tier2MonthlyEexEvaluationError("fold delivery grid is empty")
    contiguous = pd.period_range(delivery_periods[0], delivery_periods[-1], freq="M")
    return historical, retained, target_identity, tuple(str(month) for month in contiguous)


def _ordered_base_hashes(
    frame: pd.DataFrame,
    *,
    label: str,
) -> tuple[str, ...]:
    ordered = frame.sort_values(
        ["available_at", "snapshot_id", "market", "load_type", "product", "row_hash"],
        kind="mergesort",
    )
    base = ordered[
        ordered["market"].astype(str).eq("CH")
        & ordered["load_type"].astype(str).eq("BASE")
    ]
    selected = tuple(base["row_hash"].astype(str))
    if len(selected) != len(set(selected)):
        raise Tier2MonthlyEexEvaluationError(f"{label} contains duplicates")
    return selected


def _verified_plan_payload(plan: object) -> Mapping[str, object]:
    if type(plan) is not VerifiedEvaluationPlan:
        raise Tier2MonthlyEexEvaluationError(
            "selection input derivation requires a path-verified plan"
        )
    payload = getattr(plan, "payload", None)
    if not isinstance(payload, Mapping):
        raise Tier2MonthlyEexEvaluationError("verified plan payload is unavailable")
    for attribute in ("plan_id", "artifact_sha256", "trusted_time_utc"):
        if not getattr(plan, attribute, None):
            raise Tier2MonthlyEexEvaluationError(
                "verified plan identity is unavailable"
            )
    return payload


def _preflight_selection_folds(
    plan_payload: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    raw_folds = plan_payload.get("selection_folds")
    if not isinstance(raw_folds, Sequence) or isinstance(raw_folds, (str, bytes)):
        raise Tier2MonthlyEexEvaluationError(
            "verified plan selection folds are unavailable"
        )
    if not raw_folds or len(raw_folds) > MAX_SELECTION_FOLDS:
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION fold count exceeds governed resource bounds"
        )
    if any(not isinstance(fold, Mapping) for fold in raw_folds):
        raise Tier2MonthlyEexEvaluationError("SELECTION fold is not an object")
    folds = tuple(fold for fold in raw_folds if isinstance(fold, Mapping))
    if any(
        str(fold.get("target_load_type", "")) != "BASE" for fold in folds
    ):
        raise Tier2MonthlyEexEvaluationError(UNSUPPORTED_NON_BASE_SELECTION)
    fold_ids = tuple(str(fold.get("fold_id", "")) for fold in folds)
    if any(not fold_id for fold_id in fold_ids) or len(set(fold_ids)) != len(fold_ids):
        raise Tier2MonthlyEexEvaluationError(
            "SELECTION fold inventory is incomplete or duplicated"
        )
    return folds


__all__: list[str] = []
