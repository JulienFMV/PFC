from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.evaluation_curve_assembly import (
    CurveAssemblyAuthority,
    CurveAssemblyError,
    assemble_evaluation_curves,
)
from pfc_shaping.lt.evaluation_engine import SyntheticEvaluationSet
from pfc_shaping.lt.evaluation_factor_postprocess import postprocess_challenger_factors
from pfc_shaping.lt.evaluation_protocol import CandidateRole, default_evaluation_protocol
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP

ROOT = Path(__file__).resolve().parents[1]


class _FixedPredictor:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = np.array(raw, copy=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert len(features) == len(self.raw)
        return np.array(self.raw, copy=True)


class _ContextIntraday:
    def apply(
        self,
        idx: pd.DatetimeIndex,
        cal: pd.DataFrame,
        entso_forecast: pd.DataFrame | None = None,
        reference_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        del cal, reference_date
        assert entso_forecast is not None
        aligned = entso_forecast.reindex(idx)
        solar = aligned["solar_regime"].to_numpy(dtype=float)
        quarter = idx.minute.to_numpy() // 15
        raw = 1.0 + (quarter - 1.5) * 0.04 * solar
        factors = pd.Series(raw, index=idx, dtype=float)
        return (factors / factors.groupby(idx.floor("h")).transform("mean")).rename("f_Q")


class _ContextWaterValue:
    enforce_floor = False

    def compute_delta_wv(
        self,
        base: pd.Series,
        *,
        fill_df: pd.DataFrame | None,
        calendar_df: pd.DataFrame,
    ) -> pd.Series:
        del calendar_df
        assert fill_df is not None
        fill = fill_df.reindex(base.index)["fill_deviation"].to_numpy(dtype=float)
        return pd.Series(0.01 * base.abs().to_numpy(dtype=float) * fill, index=base.index)


def _origin():
    return default_evaluation_protocol().holdout.origin_slots[0]


def _delivery() -> pd.DatetimeIndex:
    local = pd.date_range(
        "2026-11-01",
        "2026-12-01",
        freq="15min",
        inclusive="left",
        tz="Europe/Zurich",
    )
    return local.tz_convert("UTC")


def _raw_native() -> np.ndarray:
    phase = np.linspace(0.0, 8.0 * np.pi, len(_delivery()), endpoint=False)
    return 1.0 + 0.25 * np.sin(phase)


def _template(**overrides) -> PFCAssembler:
    incumbent = ShapeHourlyMLP()
    incumbent.mlp_ = _FixedPredictor(_raw_native())  # type: ignore[assignment]
    incumbent.f_W_ = {
        "Ouvrable": 1.08,
        "Samedi": 0.94,
        "Dimanche": 0.82,
        "Ferie_CH": 0.79,
    }
    kwargs = {
        "shape_hourly": incumbent,
        "shape_intraday": _ContextIntraday(),
        "monthly_level_authority": "solver",
        "skip_legacy_level_cascade": True,
        "skip_legacy_base_smoothing": True,
        "calibrator": None,
        "water_value": _ContextWaterValue(),
        "uncertainty": None,
    }
    kwargs.update(overrides)
    return PFCAssembler(**kwargs)


def _challenger_batches(delivery: pd.DatetimeIndex | None = None):
    delivery = _delivery() if delivery is None else delivery
    origin = _origin()
    challenger_ids = [
        item.candidate_id
        for item in default_evaluation_protocol().candidates
        if item.role is CandidateRole.CHALLENGER
    ]
    raws = (
        _raw_native(),
        1.0 + 0.18 * np.cos(np.linspace(0.0, 5.0 * np.pi, len(delivery))),
        np.linspace(0.65, 1.35, len(delivery)),
        np.where(np.arange(len(delivery)) % 8 < 4, 0.75, 1.25),
    )
    return {
        candidate_id: postprocess_challenger_factors(
            raw,
            delivery,
            candidate_id=candidate_id,
            origin_slot_id=origin.slot_id,
            origin_as_of_utc=origin.origin_as_of_utc,
        )
        for candidate_id, raw in zip(challenger_ids, raws, strict=True)
    }


def _context() -> pd.DataFrame:
    delivery = _delivery()
    return pd.DataFrame(
        {
            "solar_regime": np.where(delivery.hour.to_numpy() < 12, 0.8, 1.2),
            "load_deviation": np.zeros(len(delivery)),
            "flow_deviation": np.zeros(len(delivery)),
        },
        index=delivery,
    )


def _hydro_context() -> pd.DataFrame:
    delivery = _delivery()
    return pd.DataFrame(
        {"fill_deviation": np.sin(np.linspace(0.0, 2.0 * np.pi, len(delivery)))},
        index=delivery,
    )


def _assemble(template: PFCAssembler | None = None, batches=None):
    return assemble_evaluation_curves(
        _template() if template is None else template,
        _challenger_batches() if batches is None else batches,
        base_prices={"2026-11": 100.0, "2026-11-Peak": 118.0},
        quoted_keys={"2026-11", "2026-11-Peak"},
        entso_forecast=_context(),
        hydro_forecast=_hydro_context(),
    )


def test_all_five_candidates_share_native_full_price_assembly() -> None:
    template = _template()
    assert not hasattr(template, "final_product_projection_report_")

    observed = _assemble(template)

    assert tuple(observed.predictions_eur_mwh) == tuple(
        item.candidate_id for item in default_evaluation_protocol().candidates
    )
    np.testing.assert_array_equal(
        observed.predictions_eur_mwh["current-unweighted-mlp"],
        observed.predictions_eur_mwh["recency-weighted-mlp"],
    )
    local = observed.delivery_at_utc.tz_convert("Europe/Zurich")
    peak = (local.weekday < 5) & (local.hour >= 8) & (local.hour < 20)
    for prediction in observed.predictions_eur_mwh.values():
        assert prediction.flags.writeable is False
        assert float(prediction.mean()) == pytest.approx(100.0, abs=1e-9)
        assert float(prediction[peak].mean()) == pytest.approx(118.0, abs=1e-9)
    assert set(observed.audit["common_component_sha256"]) == {
        "B",
        "f_S",
        "f_W",
        "f_Q",
        "f_WV",
        "delta_wv",
        "f_bridge",
    }
    assert observed.audit["monthly_level_authority"] == "CH_MONTHLY_BASE_SOLVER_UNCHANGED"
    assert observed.authority == CurveAssemblyAuthority()
    assert not any(observed.authority.to_manifest().values())
    assert not hasattr(template, "final_product_projection_report_")
    with pytest.raises(TypeError):
        observed.audit["common_component_sha256"]["f_W"] = "changed"
    assert (
        observed.to_manifest()["common_component_sha256"]["f_W"]
        == observed.audit["common_component_sha256"]["f_W"]
    )
    scoring_input = SyntheticEvaluationSet(
        fixture_id="synthetic-common-assembly",
        origin_slot_id=observed.origin_slot_id,
        origin_as_of_utc=observed.origin_as_of_utc,
        delivery_at_utc=observed.delivery_at_utc,
        truth_eur_mwh=observed.predictions_eur_mwh["current-unweighted-mlp"],
        predictions_eur_mwh=observed.predictions_eur_mwh,
    )
    assert tuple(scoring_input.predictions_eur_mwh) == tuple(observed.predictions_eur_mwh)
    with pytest.raises(FrozenInstanceError):
        observed.authority.production_authorized = True  # type: ignore[misc]


@pytest.mark.parametrize(
    ("template", "message"),
    [
        (_template(monthly_level_authority="legacy"), "solver monthly-level authority"),
        (_template(skip_legacy_level_cascade=False), "legacy level paths disabled"),
        (_template(enable_solar_modulation=True), "optional assembly layers"),
    ],
)
def test_assembly_rejects_noncanonical_templates(
    template: PFCAssembler,
    message: str,
) -> None:
    with pytest.raises(CurveAssemblyError, match=message):
        _assemble(template)


def test_assembly_requires_exact_hashed_challenger_inventory() -> None:
    batches = _challenger_batches()
    batches.pop("ridge-linear")
    with pytest.raises(CurveAssemblyError, match="exact frozen inventory"):
        _assemble(batches=batches)

    batches = _challenger_batches()
    original = batches["ridge-linear"]
    changed = np.array(original.values, copy=True)
    changed[0] += 0.01
    changed.setflags(write=False)
    batches["ridge-linear"] = replace(original, values=changed)
    with pytest.raises(CurveAssemblyError, match="factor hash"):
        _assemble(batches=batches)


def test_curve_assembly_has_no_io_fit_score_ct_or_gpu_path() -> None:
    source = (
        (ROOT / "pfc_shaping/lt/evaluation_curve_assembly.py").read_text(encoding="utf-8").lower()
    )
    forbidden = (
        "pfc_shaping.ct",
        "read_parquet",
        "read_csv",
        "to_parquet",
        "to_csv",
        "requests.",
        ".fit(",
        "evaluate_synthetic_predictions",
        "cuda",
    )
    assert not any(fragment in source for fragment in forbidden)
