"""Pure common full-price assembly for frozen LT hourly candidates.

The native incumbent keeps its own ``ShapeHourlyMLP.apply`` path.  Challengers
replace only that one output seam with already postprocessed ``f_H`` values;
the existing solver-authority assembler remains the single implementation of
all downstream curve layers.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.lt.evaluation_factor_postprocess import PostprocessedChallengerFactors
from pfc_shaping.lt.evaluation_protocol import CandidateRole, default_evaluation_protocol
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP

CURVE_ASSEMBLY_SCHEMA = "fmv-lt-common-full-price-assembly.v1"
INCUMBENT_CANDIDATE_ID = "current-unweighted-mlp"
_COMMON_COMPONENTS = ("B", "f_S", "f_W", "f_Q", "f_WV", "delta_wv", "f_bridge")


class CurveAssemblyError(ValueError):
    """Raised when a candidate cannot enter the common assembly boundary."""


@dataclass(frozen=True, slots=True)
class CurveAssemblyAuthority:
    """Non-overridable negative authority for assembled evaluation curves."""

    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class AssembledEvaluationCurves:
    """Detached EUR/MWh predictions for the exact frozen candidate inventory."""

    origin_slot_id: str
    origin_as_of_utc: datetime
    delivery_at_utc: pd.DatetimeIndex
    predictions_eur_mwh: Mapping[str, np.ndarray]
    audit: Mapping[str, object]
    authority: CurveAssemblyAuthority = field(default_factory=CurveAssemblyAuthority, init=False)

    def to_manifest(self) -> dict[str, object]:
        audit = dict(self.audit)
        for key in (
            "common_component_sha256",
            "prediction_sha256",
            "challenger_factor_sha256",
        ):
            audit[key] = dict(audit[key])
        audit["candidate_ids"] = list(audit["candidate_ids"])
        audit["common_layers"] = list(audit["common_layers"])
        return {
            "schema_version": CURVE_ASSEMBLY_SCHEMA,
            "status": "PASS_COMMON_FULL_PRICE_ASSEMBLY_NO_OPERATIONAL_AUTHORITY",
            "origin_slot_id": self.origin_slot_id,
            "origin_as_of_utc": self.origin_as_of_utc.isoformat(),
            "delivery_grain": "NATIVE_QUARTER_HOUR_UTC_GRID",
            "candidate_output": "FULL_PRICE_EUR_MWH",
            **audit,
            "authority": self.authority.to_manifest(),
        }


class _InjectedHourlyFactors:
    """Minimal ShapeHourly view that changes only ``apply`` output."""

    def __init__(
        self,
        incumbent: ShapeHourlyMLP,
        delivery_at_utc: pd.DatetimeIndex,
        values: np.ndarray,
    ) -> None:
        self._delivery_at_utc = delivery_at_utc.copy()
        self._values = _readonly(values)
        self._use_seasonal_hourly = bool(getattr(incumbent, "_use_seasonal_hourly", False))
        self.f_W_ = MappingProxyType(dict(incumbent.f_W_))
        self.f_W_seasonal_ = MappingProxyType(dict(incumbent.f_W_seasonal_))

    def apply(
        self,
        timestamps: pd.DatetimeIndex,
        calendar_df: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
        outages_forecast: pd.DataFrame | None = None,
    ) -> pd.Series:
        del calendar_df, reference_date, outages_forecast
        observed = _utc_index(timestamps, label="assembler delivery")
        if not observed.equals(self._delivery_at_utc):
            raise CurveAssemblyError("assembler requested a different challenger delivery grid")
        return pd.Series(np.array(self._values, copy=True), index=observed, name="f_H")


def assemble_evaluation_curves(
    assembler_template: PFCAssembler,
    challenger_factors: Mapping[str, PostprocessedChallengerFactors],
    *,
    base_prices: Mapping[str, float],
    quoted_keys: set[str] | frozenset[str],
    entso_forecast: pd.DataFrame | None = None,
    hydro_forecast: pd.DataFrame | None = None,
) -> AssembledEvaluationCurves:
    """Assemble the native incumbent and four challengers on one common path."""

    _validate_template(assembler_template)
    protocol = default_evaluation_protocol()
    expected_candidates = tuple(item.candidate_id for item in protocol.candidates)
    expected_challengers = tuple(
        item.candidate_id for item in protocol.candidates if item.role is CandidateRole.CHALLENGER
    )
    if not isinstance(challenger_factors, Mapping) or set(challenger_factors) != set(
        expected_challengers
    ):
        raise CurveAssemblyError("challenger factors must match the exact frozen inventory")

    batches = tuple(challenger_factors[candidate_id] for candidate_id in expected_challengers)
    for candidate_id, batch in zip(expected_challengers, batches, strict=True):
        _validate_factor_batch(batch, expected_candidate_id=candidate_id)
    first = batches[0]
    delivery = _utc_index(first.delivery_at_utc, label="challenger delivery")
    origin = pd.Timestamp(first.origin_as_of_utc).tz_convert("UTC")
    for batch in batches[1:]:
        if (
            batch.origin_slot_id != first.origin_slot_id
            or batch.origin_as_of_utc != first.origin_as_of_utc
        ):
            raise CurveAssemblyError("challenger factors do not share one frozen origin")
        if not delivery.equals(_utc_index(batch.delivery_at_utc, label="challenger delivery")):
            raise CurveAssemblyError("challenger factors do not share one delivery grid")

    slot = next(
        (item for item in protocol.holdout.origin_slots if item.slot_id == first.origin_slot_id),
        None,
    )
    if slot is None or slot.origin_as_of_utc != first.origin_as_of_utc:
        raise CurveAssemblyError("factor origin is not the exact frozen protocol slot")
    if not delivery.is_monotonic_increasing or delivery.has_duplicates:
        raise CurveAssemblyError("delivery grid must be unique and strictly increasing")
    if len(delivery) < 2 or not bool(
        (delivery.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    ):
        raise CurveAssemblyError("delivery grid must have exact 15-minute cadence")

    prices = _validated_prices(base_prices, delivery)
    accepted_keys = _validated_quoted_keys(quoted_keys, prices)
    shared_inputs = {
        "base_prices": prices,
        "quoted_keys": accepted_keys,
        "delivery_index": delivery,
        "entso_forecast": _detached_frame(entso_forecast),
        "hydro_forecast": _detached_frame(hydro_forecast),
        "outages_forecast": None,
        "reference_date": origin,
        "country": "CH",
    }

    assembled: dict[str, pd.DataFrame] = {}
    assembled[INCUMBENT_CANDIDATE_ID] = _build_from_template(
        assembler_template,
        assembler_template.sh,
        shared_inputs,
    )
    for candidate_id, batch in zip(expected_challengers, batches, strict=True):
        injected = _InjectedHourlyFactors(assembler_template.sh, delivery, batch.values)
        assembled[candidate_id] = _build_from_template(
            assembler_template,
            injected,
            shared_inputs,
        )

    incumbent_frame = assembled[INCUMBENT_CANDIDATE_ID]
    _validate_common_assembly(assembled, incumbent_frame, delivery)
    predictions = {
        candidate_id: _readonly(assembled[candidate_id]["price_shape"].to_numpy(dtype=float))
        for candidate_id in expected_candidates
    }
    component_hashes = MappingProxyType(
        {
            name: _array_hash(incumbent_frame[name].to_numpy(dtype=float))
            for name in _COMMON_COMPONENTS
        }
    )
    audit = MappingProxyType(
        {
            "candidate_ids": expected_candidates,
            "row_count": len(delivery),
            "delivery_sha256": _timestamp_hash(delivery),
            "solver_level_sha256": component_hashes["B"],
            "common_component_sha256": component_hashes,
            "prediction_sha256": MappingProxyType(
                {candidate_id: _array_hash(values) for candidate_id, values in predictions.items()}
            ),
            "challenger_factor_sha256": MappingProxyType(
                {batch.candidate_id: str(batch.audit["value_sha256"]) for batch in batches}
            ),
            "incumbent_path": "NATIVE_SHAPE_HOURLY_MLP_APPLY",
            "challenger_override_seam": "POSTPROCESSED_F_H_ONLY",
            "monthly_level_authority": "CH_MONTHLY_BASE_SOLVER_UNCHANGED",
            "common_layers": (
                "INCUMBENT_F_W",
                "QUARTER_HOUR_F_Q_WITH_SHARED_CONTEXT",
                "WATER_VALUE_F_WV_OR_DELTA",
                "HORIZON_DAMPING",
                "NEAR_TERM_BRIDGE",
                "MONTHLY_SOLVER_RECENTERING",
                "FINAL_BASE_PEAK_OFFPEAK_PROJECTION",
            ),
            "model_fit_performed": False,
            "truth_opened": False,
            "scoring_performed": False,
            "ranking_or_selection_performed": False,
            "warehouse_start_count": 0,
            "gpu_execution_count": 0,
        }
    )
    return AssembledEvaluationCurves(
        origin_slot_id=first.origin_slot_id,
        origin_as_of_utc=first.origin_as_of_utc,
        delivery_at_utc=delivery,
        predictions_eur_mwh=MappingProxyType(predictions),
        audit=audit,
    )


def _validate_template(template: object) -> None:
    if not isinstance(template, PFCAssembler):
        raise CurveAssemblyError("assembler_template must be a PFCAssembler")
    if type(template.sh) is not ShapeHourlyMLP:
        raise CurveAssemblyError(
            "assembler_template must retain the native ShapeHourlyMLP incumbent"
        )
    if str(template.monthly_level_authority).lower() != "solver":
        raise CurveAssemblyError("assembler_template must retain solver monthly-level authority")
    if not template.skip_legacy_level_cascade or not template.skip_legacy_base_smoothing:
        raise CurveAssemblyError("solver assembly must keep both legacy level paths disabled")
    if template.unc is not None:
        raise CurveAssemblyError("uncertainty bands are outside deterministic evaluation assembly")
    forbidden_flags = (
        "enable_solar_modulation",
        "enable_electrification_shape",
        "enable_intraday_amplitude_shrinkage",
    )
    enabled = [name for name in forbidden_flags if bool(getattr(template, name, False))]
    if enabled:
        raise CurveAssemblyError(f"unfrozen optional assembly layers are enabled: {enabled}")


def _validate_factor_batch(
    batch: object,
    *,
    expected_candidate_id: str,
) -> None:
    if not isinstance(batch, PostprocessedChallengerFactors):
        raise CurveAssemblyError("each challenger value must be a postprocessed factor batch")
    if batch.candidate_id != expected_candidate_id:
        raise CurveAssemblyError("challenger factor key and candidate_id differ")
    if any(batch.authority.to_manifest().values()):
        raise CurveAssemblyError("challenger factor batch carries forbidden authority")
    delivery = _utc_index(batch.delivery_at_utc, label="challenger delivery")
    values = np.asarray(batch.values, dtype=float)
    if values.shape != (len(delivery),) or not np.isfinite(values).all():
        raise CurveAssemblyError("challenger factors must be one finite value per delivery row")
    if values.flags.writeable:
        raise CurveAssemblyError("challenger factors must be read-only")
    if batch.audit.get("delivery_sha256") != _timestamp_hash(delivery):
        raise CurveAssemblyError("challenger delivery hash does not match its values")
    if batch.audit.get("value_sha256") != _array_hash(values):
        raise CurveAssemblyError("challenger factor hash does not match its values")


def _build_from_template(
    template: PFCAssembler,
    hourly_shape: object,
    shared_inputs: Mapping[str, object],
) -> pd.DataFrame:
    worker = copy.copy(template)
    worker.sh = hourly_shape
    worker._sh_accepts_outages = (
        template._sh_accepts_outages if hourly_shape is template.sh else True
    )
    kwargs = dict(shared_inputs)
    for name in ("entso_forecast", "hydro_forecast", "outages_forecast"):
        kwargs[name] = _detached_frame(kwargs[name])
    frame = worker.build(**kwargs)
    if not frame.index.equals(shared_inputs["delivery_index"]):
        raise CurveAssemblyError("assembler output changed the frozen delivery grid")
    return frame


def _validate_common_assembly(
    assembled: Mapping[str, pd.DataFrame],
    incumbent: pd.DataFrame,
    delivery: pd.DatetimeIndex,
) -> None:
    for candidate_id, frame in assembled.items():
        if not frame.index.equals(delivery):
            raise CurveAssemblyError(f"{candidate_id} output is not on the common delivery grid")
        if (
            "price_shape" not in frame
            or not np.isfinite(frame["price_shape"].to_numpy(dtype=float)).all()
        ):
            raise CurveAssemblyError(f"{candidate_id} did not produce a finite EUR/MWh curve")
        if "calibrated" not in frame or not bool(frame["calibrated"].astype(bool).all()):
            raise CurveAssemblyError(f"{candidate_id} skipped final solver product projection")
        month_key = pd.Index(
            delivery.tz_convert("Europe/Zurich").strftime("%Y-%m"),
            name="month_key",
        )
        price_means = frame["price_shape"].groupby(month_key).mean().to_numpy(dtype=float)
        level_means = frame["B"].groupby(month_key).mean().to_numpy(dtype=float)
        if not np.allclose(price_means, level_means, rtol=0.0, atol=1e-6):
            raise CurveAssemblyError(f"{candidate_id} changed solver monthly means")
        for name in _COMMON_COMPONENTS:
            if name not in frame or not np.array_equal(
                frame[name].to_numpy(dtype=float),
                incumbent[name].to_numpy(dtype=float),
            ):
                raise CurveAssemblyError(f"{candidate_id} changed common assembly component {name}")


def _validated_prices(values: object, delivery: pd.DatetimeIndex) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise CurveAssemblyError("base_prices must be a non-empty mapping")
    result: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise CurveAssemblyError(f"base price is not numeric: {key}") from exc
        if not key or not np.isfinite(value):
            raise CurveAssemblyError("base price keys and values must be finite")
        result[key] = value
    if not any(len(key) == 7 and key[4] == "-" and key[5:].isdigit() for key in result):
        raise CurveAssemblyError("solver assembly requires explicit monthly BASE levels")
    missing = set(delivery.tz_convert("Europe/Zurich").strftime("%Y-%m")) - result.keys()
    if missing:
        raise CurveAssemblyError(f"explicit solver BASE required for every delivery month: {sorted(missing)}")
    return result


def _validated_quoted_keys(values: object, prices: Mapping[str, float]) -> set[str]:
    if not isinstance(values, (set, frozenset)):
        raise CurveAssemblyError("quoted_keys must be an explicit set")
    result = {str(value) for value in values}
    if not result.issubset(prices):
        raise CurveAssemblyError("quoted_keys must refer to supplied base_prices")
    return result


def _detached_frame(value: object) -> pd.DataFrame | None:
    if value is None:
        return None
    if not isinstance(value, pd.DataFrame):
        raise CurveAssemblyError("common context inputs must be DataFrames or None")
    return value.copy(deep=True)


def _utc_index(value: object, *, label: str) -> pd.DatetimeIndex:
    if not isinstance(value, pd.DatetimeIndex) or value.tz is None:
        raise CurveAssemblyError(f"{label} must be a timezone-aware DatetimeIndex")
    return value.tz_convert("UTC").copy()


def _readonly(values: object) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_hash(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {"dtype": "<f8", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + canonical.tobytes(order="C")).hexdigest()


def _timestamp_hash(values: pd.DatetimeIndex) -> str:
    payload = json.dumps(
        [value.isoformat() for value in values],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CURVE_ASSEMBLY_SCHEMA",
    "AssembledEvaluationCurves",
    "CurveAssemblyAuthority",
    "CurveAssemblyError",
    "assemble_evaluation_curves",
]
