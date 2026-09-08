"""Execute the fixed local D305 hydro/disaggregation/PFC composition experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from pfc_shaping.data.lt_replay_transforms import build_hydro_water_value
from pfc_shaping.lt.local_benchmark import AUTHORITIES, CORRECTED, score_curves
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from pfc_shaping.lt.model.water_value import WaterValueCorrection
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference, closed_month_targets
from pfc_shaping.lt.signed_intraday import intraday_cell_reference
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from pfc_shaping.pipeline.production_phases import _future_hydro_civil_weekly_index
from pfc_shaping.validation.product_normalization import build_product_normalization_gates

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build/lt-signed-composition-20260907"
OLD = ROOT / "build/local-lt-benchmark-20260907"
SIGNED = ROOT / "build/lt-signed-benchmark-20260907/run-v1"
SOURCE = ROOT / "build/local-pfc-source-preflight-20260907"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def assembler(hourly, intraday=None, water=None):
    return PFCAssembler(hourly, intraday if intraday is not None else ShapeIntraday(),
        uncertainty=None, water_value=water, cascader=None,
        calibrator=ArbitrageFreeCalibrator(smoothness_weight=1., tol=.01),
        calibration_fallback_to_raw=False, peak_source_policy="same_first",
        monthly_level_authority="solver", skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True, monthly_constraint_tolerance=1e-9)


def save_curve(folder, frame, base_prices, surface, assembly):
    folder.mkdir()
    if not frame.index.is_unique or not np.isfinite(frame.price_shape).all():
        raise ValueError("invalid assembled curve")
    monthly = frame.price_shape.groupby(frame.index.tz_convert("Europe/Zurich").strftime("%Y-%m")).mean()
    residual = monthly - pd.Series(base_prices).reindex(monthly.index)
    if not np.isfinite(residual).all() or residual.abs().max() > 1e-9:
        raise ValueError("monthly solver level changed")
    hourly = frame.price_shape.resample("h").mean().to_frame("price_eur_mwh")
    hourly["ts_ch"] = hourly.index.tz_convert("Europe/Zurich")
    for field in ("year", "month", "quarter"):
        hourly[field] = getattr(hourly.ts_ch.dt, field)
    gates = build_product_normalization_gates(hourly, surface,
        forward_date=pd.Timestamp(surface.date.iloc[0]), price_column="price_eur_mwh",
        hard_tolerance=1e-6, peak_country="CH")
    if gates.status.eq("CRITICAL").any():
        raise ValueError("critical product gate")
    gates.to_csv(folder / "product-gates.csv", index=False)
    frame.to_parquet(folder / "curve.parquet")
    receipt = dict(rows=len(frame), monthly_max_error=float(residual.abs().max()),
        product_gate_counts={str(k): int(v) for k, v in gates.status.value_counts().items()},
        projection=assembly.final_product_projection_report_, authority=dict(AUTHORITIES))
    write_json(folder / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd() != ROOT or ROOT != Path(r"C:\Users\jbattaglia\PFC_LT"):
        raise RuntimeError("canonical workspace required")
    for name in ("TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "MPLCONFIGDIR", "XDG_CACHE_HOME",
                 "NUMBA_CACHE_DIR", "JOBLIB_TEMP_FOLDER", "PYTHONUSERBASE", "PIP_CACHE_DIR"):
        if not Path(os.environ.get(name, "")).resolve().is_relative_to(WORK):
            raise RuntimeError(f"task-local runtime required: {name}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU-only execution required")
    out = args.output.resolve()
    if not out.is_relative_to(WORK) or out == WORK:
        raise ValueError("fresh output below composition task root required")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=out / "runtime.log", level=logging.INFO)
    inputs = {}

    def checked(path, digest):
        if sha(path) != digest:
            raise RuntimeError(f"input mismatch: {path}")
        inputs[path.relative_to(ROOT).as_posix()] = digest
        return path

    inventory = json.loads(checked(OLD / "session-artifacts.json",
        "3a0ac1e3dcdc9777c4a2f75bf82a606a0b3b82f5cd66f34946dc09c3ce6dde1a").read_text())
    signed_manifest = json.loads(checked(SIGNED / "manifest.json",
        "446c940cc1a62845bbde664bdc61509078efca802467ccfd7451982031f134af").read_text())

    def old(name):
        return checked(OLD / name, inventory["files"][name]["sha256"])

    def signed(name):
        return checked(SIGNED / name, signed_manifest[name])

    baseline = json.loads(checked(WORK / "baseline/manifest.json",
        "c0e3705a9f6b1dfc7a20a6955e1d04e7a57e921fa4375c8777a4bb614114b8fc").read_text())
    code = {}
    changed = {"pfc_shaping/lt/model/assembler.py", "tests/test_signed_hourly_assembly.py"}
    for name, digest in baseline.items():
        checked(WORK / "baseline" / name, digest)
        code[name] = sha(ROOT / name)
        if code[name] != digest and name not in changed:
            raise RuntimeError(f"unexpected dependency change: {name}")
    for name in ("scripts/run_lt_signed_composition.py", "pfc_shaping/lt/signed_intraday.py",
                 "tests/test_signed_composition.py", "docs/model/LT-SIGNED-COMPOSITION-LOCAL-EXPERIMENT.md"):
        code[name] = sha(ROOT / name)
    prep = json.loads(checked(SOURCE / "prepared-inputs/manifest.json",
        "36f27a705bdf56ff723d778d226ed3b5f4ae594f20e54b402cc3d6385088b3ef").read_text())
    for name, digest in prep["files"].items():
        checked(SOURCE / "prepared-inputs" / name, digest)
    fitted = json.loads(checked(SOURCE / "fitted-models/manifest.json",
        "209a427339159a0e39ca9c7705ad55f522ceb943534f6038efade157b842c5ca").read_text())
    for name, digest in fitted["files"].items():
        checked(SOURCE / "fitted-models" / name, digest)
    current = json.loads(checked(SOURCE / "curve-final/manifest.json",
        "ec9680351a79fac993f12b6c023e5bbba194ca887eff885dc02feee4159acae4").read_text())
    for name, digest in current["files"].items():
        checked(SOURCE / "curve-final" / name, digest)
    # These two existing helpers are new dependencies relative to D304. Bind
    # their code to the already frozen D300 source receipts before execution.
    for name, digest in (
        ("pfc_shaping/data/lt_replay_transforms.py", prep["code_sha256"]["pfc_shaping/data/lt_replay_transforms.py"]),
        ("pfc_shaping/pipeline/production_phases.py", current["source_code_sha256"]["pfc_shaping/pipeline/production_phases.py"]),
    ):
        checked(ROOT / name, digest)
        code[name] = digest
    checked(SOURCE / "monthly-solver/result.json", current["solver_result_sha256"])
    # D301 already binds the retained normalized EEX history.
    old_plan = json.loads(old("plan.json").read_text())
    eex_path = SOURCE / "eex-replay/eex-normalized-history.parquet"
    checked(eex_path, old_plan["sources_and_code_sha256"][eex_path.relative_to(ROOT).as_posix()])
    for spec in old_plan["folds"]:
        for name in ("training-hydro.parquet", "solver.json", "common-native-curve.parquet",
                     "eex-surface.parquet", f"{CORRECTED}.pkl"):
            old(f"{spec['id']}/{name}")
        for name in ("targets.parquet", "signed-seasonal/curve.parquet"):
            signed(f"{spec['id']}/{name}")
    plan = dict(schema="fmv-lt-signed-composition-local.v1",
        frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(), folds=old_plan["folds"],
        de_months=[f"2026-{month:02d}" for month in range(1, 9)],
        de_candidates=["flat", "native", "native-regularized", "signed-additive"],
        hydro_candidates=["mlp", "mlp-hydro", "signed", "signed-hydro"],
        current_candidates=["mlp-current", "signed", "signed-hydro", "signed-intraday", "signed-both"],
        source_policy="LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT",
        comparison_policy="EXPOSED_LOCAL_DEVELOPMENT_NO_PROMOTION",
        code_sha256=code, inputs_sha256=dict(inputs), authority=dict(AUTHORITIES))
    write_json(out / "plan.json", plan)
    print(json.dumps(dict(stage="plan_frozen", sha256=sha(out / "plan.json"))), flush=True)
    truth = pd.read_parquet(SOURCE / "prepared-inputs/epex-ch.parquet")["price_eur_mwh"]
    if not (truth.resample("h").max() - truth.resample("h").min()).eq(0).all():
        raise ValueError("CH truth resolution changed")
    truth = truth.loc[truth.index < pd.Timestamp("2026-09-01", tz="Europe/Zurich")]
    all_metrics, controls, receipts = [], [], []
    for spec in plan["folds"]:
        year, origin = spec["id"], pd.Timestamp(spec["origin_utc"])
        folder = out / year
        folder.mkdir()
        targets = pd.read_parquet(signed(f"{year}/targets.parquet"))
        hydro = build_hydro_water_value(pd.read_parquet(old(f"{year}/training-hydro.parquet")))
        if hydro.index.max() >= origin or targets.index.max() >= origin:
            raise ValueError("future input in historical fit")
        # ISO week53 can lack five prior analogues. Carry the latest supported
        # pre-origin anomaly, with a bounded age; never treat its neutral sentinel
        # as an observation or use a future week to reconstruct it.
        hydro = hydro.loc[hydro.water_value_supported].copy()
        if hydro.empty or origin - hydro.index[-1] > pd.Timedelta(days=14):
            raise ValueError("no supported pre-origin hydro anomaly within 14 days")
        hydro.to_parquet(folder / "training-hydro-derived.parquet")
        # Reuse existing water fit; complete pre-origin months only.
        water = WaterValueCorrection().fit(targets[["price_eur_mwh"]], hydro, enrich_15min_index(targets.index))
        if water.n_obs_ < 12:
            raise ValueError("water-value fit fell back without sufficient observations")
        water.save(folder / "water-value.parquet")
        native_saved = pd.read_parquet(old(f"{year}/common-native-curve.parquet"))
        signed_saved = pd.read_parquet(signed(f"{year}/signed-seasonal/curve.parquet"))
        grid = native_saved.index
        future_weeks = _future_hydro_civil_weekly_index(hydro.index[-1], grid[-1] + pd.Timedelta(days=8))
        forecast = pd.DataFrame({"fill_deviation": np.linspace(float(hydro.fill_deviation.iloc[-1]), 0., len(future_weeks))}, index=future_weeks)
        forecast.to_parquet(folder / "hydro-forecast.parquet")
        write_json(folder / "hydro-fit.json", dict(observations=water.n_obs_,
            beta=water.beta_wv_, last_training_price=targets.index[-1].isoformat(),
            last_training_hydro=hydro.index[-1].isoformat(), origin=origin.isoformat()))
        with old(f"{year}/{CORRECTED}.pkl").open("rb") as stream:
            hourly = pickle.load(stream)
        solver = json.loads(old(f"{year}/solver.json").read_text())
        shared = dict(base_prices=solver["base_prices"], quoted_keys=set(solver["quoted_keys"]),
            delivery_index=grid, reference_date=origin, country="CH")
        surface = pd.read_parquet(old(f"{year}/eex-surface.parquet"))
        shape = signed_saved.signed_hourly_shape_eur_mwh.iloc[::4]
        for candidate in plan["hydro_candidates"]:
            use_water = candidate.endswith("hydro")
            assembly = assembler(hourly, water=water if use_water else None)
            frame = assembly.build(**shared,
                signed_hourly_shape=shape if candidate.startswith("signed") else None,
                hydro_forecast=forecast if use_water else None)
            receipt = save_curve(folder / candidate, frame, shared["base_prices"], surface, assembly)
            receipts.append(dict(origin=year, candidate=candidate, **receipt))
            if not use_water:
                control = signed_saved if candidate == "signed" else native_saved
                error = float((frame.price_shape - control.price_shape).abs().max())
                if error > 1e-10:
                    raise ValueError("D304 control reproduction failed")
                controls.append(dict(origin=year, candidate=candidate, max_error=error))
            for stage, column in (("final", "price_shape"), ("pre_projection", "price_pre_final_projection")):
                scores, errors, _ = score_curves(frame[[column]].rename(columns={column: candidate}), truth, origin)
                errors.to_parquet(folder / candidate / f"errors-{stage}.parquet")
                scores["origin"], scores["role"], scores["stage"] = year, spec["role"], stage
                all_metrics.append(scores)
        print(json.dumps(dict(stage="hydro_origin_complete", origin=year)), flush=True)
    pd.concat(all_metrics).to_csv(out / "hydro-metrics.csv", index=False)

    de = pd.read_parquet(SOURCE / "prepared-inputs/epex-de.parquet")
    de_scores = []
    for month in plan["de_months"]:
        first = pd.Period(month, freq="M")
        origin = first.start_time.tz_localize("Europe/Berlin")
        end = (first + 1).start_time.tz_localize("Europe/Berlin")
        train = de.loc[de.index < origin]
        test = de.loc[(de.index >= origin) & (de.index < end)]
        expected = pd.date_range(origin, end, freq="15min", inclusive="left").tz_convert("UTC")
        if not test.index.equals(expected) or train.empty or train.index.max() >= origin:
            raise ValueError("incomplete DE fold or future training")
        folder = out / f"de-{month}"
        folder.mkdir()
        cal = enrich_15min_index(test.index, country="DE")
        actual = test.price_eur_mwh
        parent = actual.groupby(actual.index.floor("h")).transform("mean")
        residual = actual - parent
        predictions = pd.DataFrame({"actual": actual, "parent_hour": parent,
            "truth_residual": residual, "flat": 0.}, index=test.index)
        for regularize in (False, True):
            model = ShapeIntraday(enable_sparse_support_regularization=regularize)
            model.fit(train, None, enrich_15min_index(train.index, country="DE"))
            name = "native-regularized" if regularize else "native"
            model.save(folder / f"{name}.parquet")
            factor = model.apply(test.index, cal, reference_date=origin)
            predictions[name] = parent * (factor - 1.)
        additive, counts = intraday_cell_reference(train.price_eur_mwh, test.index, country="DE")
        predictions["signed-additive"] = additive
        predictions.to_parquet(folder / "predictions.parquet")
        write_json(folder / "fit.json", dict(training_rows=len(train), test_rows=len(test),
            last_training_timestamp=train.index[-1].isoformat(), origin=origin.isoformat(),
            additive_fallback_counts=counts, observed_parent_hour_conditioning=True))
        for candidate in plan["de_candidates"]:
            error = predictions[candidate] - residual
            if predictions[candidate].groupby(test.index.floor("h")).mean().abs().max() > 1e-9:
                raise ValueError("DE disaggregation changed parent hour")
            for segment, mask in (("ALL", np.ones(len(test), dtype=bool)),
                                  ("NEGATIVE_PARENT", parent.to_numpy() < 0)):
                selected = error.loc[mask]
                de_scores.append(dict(month=month, candidate=candidate, segment=segment, rows=len(selected),
                    mae=float(selected.abs().mean()) if len(selected) else None,
                    rmse=float(np.sqrt(np.mean(selected ** 2))) if len(selected) else None))
        print(json.dumps(dict(stage="de_month_complete", month=month)), flush=True)
    pd.DataFrame(de_scores).to_csv(out / "de-metrics.csv", index=False)

    # Current curve generation uses exactly the retained D300 levels and contexts.
    saved = pd.read_parquet(SOURCE / "curve-final/pfc-full-native-horizon.parquet")
    grid = saved.index
    solver = json.loads((SOURCE / "monthly-solver/result.json").read_text())
    reference = pd.Timestamp(current["valuation_at_utc"])
    shared = dict(base_prices=solver["assembler_base_prices"], quoted_keys=set(solver["quoted_keys"]),
        delivery_index=grid, reference_date=reference, country="CH")
    hourly = HydroAlignedShapeHourlyMLP.load(SOURCE / "fitted-models/hourly.pkl")
    intraday = ShapeIntraday.load(SOURCE / "fitted-models/intraday.parquet")
    water = WaterValueCorrection.load(SOURCE / "fitted-models/water-value.parquet")
    forecast = pd.read_parquet(SOURCE / "curve-final/hydro-forecast.parquet")
    physical = pd.read_parquet(SOURCE / "curve-final/entsoe-climatology.parquet")
    target = closed_month_targets(truth.resample("h").mean(), reference)
    raw, counts = calendar_cell_reference(target.signed_target, grid[::4])
    shape = center_signed_hourly_shape(pd.Series(raw, index=grid[::4]))
    delta, q_counts = intraday_cell_reference(de.price_eur_mwh, grid)
    surface = select_latest_quote_surface(pd.read_parquet(eex_path))
    current_folder = out / "current"
    current_folder.mkdir()
    target.to_parquet(current_folder / "hourly-training-targets.parquet")
    write_json(current_folder / "preparation.json", dict(hourly_fallback_counts=counts,
        intraday_fallback_counts=q_counts, de_transfer_to_ch_validated=False,
        reference_utc=reference.isoformat(), authority=dict(AUTHORITIES)))
    curves = {}
    for candidate in plan["current_candidates"]:
        native = candidate == "mlp-current"
        use_water = native or candidate in ("signed-hydro", "signed-both")
        assembly = assembler(hourly, intraday if native else None, water if use_water else None)
        frame = assembly.build(**shared, entso_forecast=physical if native else None,
            hydro_forecast=forecast if use_water else None,
            signed_hourly_shape=None if native else shape,
            signed_intraday_shape=delta if candidate in ("signed-intraday", "signed-both") else None)
        receipt = save_curve(current_folder / candidate, frame, shared["base_prices"], surface, assembly)
        receipts.append(dict(origin="current", candidate=candidate, **receipt))
        if native:
            error = float((frame.price_shape - saved.price_shape).abs().max())
            if error > 1e-10:
                raise ValueError("D300 full PFC changed")
            controls.append(dict(origin="current", candidate=candidate, max_error=error))
        curves[candidate] = frame
        main = frame.loc[frame.index < pd.Timestamp("2030-01-01", tz="Europe/Zurich")]
        pd.DataFrame({"timestamp_utc": main.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timestamp_ch": main.index.tz_convert("Europe/Zurich").map(lambda t: t.isoformat()),
            "price_eur_mwh": main.price_shape.to_numpy()}).to_csv(
                current_folder / candidate / "pfc-fmv-ch-15min.csv", sep=";", index=False, float_format="%.10f")
        print(json.dumps(dict(stage="current_pfc_complete", candidate=candidate)), flush=True)
    for left, right in (("signed-intraday", "signed"), ("signed-both", "signed-hydro")):
        error = float((curves[left].price_shape - curves[right].price_shape - delta).abs().max())
        if error > 1e-9:
            raise ValueError("final projection changed additive intrahour residual")
        controls.append(dict(origin="current", candidate=f"{left}-minus-{right}", max_error=error))
    for name, digest in {**inputs, **code}.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"bound input/code changed during execution: {name}")
    write_json(out / "complete.json", dict(status="COMPLETE_LOCAL_COMPOSITION_NO_PROMOTION",
        controls=controls, receipts=receipts, hydro_fits=6, intraday_factor_fits=16,
        current_model_refits=0, completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        authority=dict(AUTHORITIES)))
    logging.shutdown()
    write_json(out / "manifest.json", {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()})


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
