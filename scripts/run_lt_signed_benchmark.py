"""Fixed local CPU representation experiment; reuses retained PRD bytes and assembly."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build/lt-signed-benchmark-20260907"
OLD = ROOT / "build/local-lt-benchmark-20260907"
SOURCE = ROOT / "build/local-pfc-source-preflight-20260907"
sys.dont_write_bytecode = True
sys.path.append(str(OLD / "dependencies"))

import numpy as np
import pandas as pd
from importlib.metadata import version

from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.evaluation_curve_assembly import _InjectedHourlyFactors
from pfc_shaping.lt.local_benchmark import (
    AUTHORITIES, CHALLENGERS, CORRECTED, FEATURES, fit_challenger,
    postprocess, predict_challenger, prediction_matrix, score_curves,
)
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.shape_hourly_mlp import _encode_features
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference, closed_month_targets
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from pfc_shaping.validation.product_normalization import build_product_normalization_gates

PARAMETERS = dict(learning_rate="0.05", min_data_in_leaf="100", n_estimators="300", num_leaves="31")
CANDIDATES = ("ratio-seasonal", "signed-seasonal", "ratio-lightgbm", "signed-lightgbm")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def make_assembler(hourly):
    return PFCAssembler(hourly, ShapeIntraday(), uncertainty=None, water_value=None,
        cascader=None, calibrator=ArbitrageFreeCalibrator(smoothness_weight=1., tol=.01),
        calibration_fallback_to_raw=False, peak_source_policy="same_first",
        monthly_level_authority="solver", skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True, monthly_constraint_tolerance=1e-9,
        allow_negative_prices=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd() != ROOT or ROOT != Path(r"C:\Users\jbattaglia\PFC_LT"):
        raise RuntimeError("canonical workstation root required")
    out = args.output.resolve()
    if not out.is_relative_to(WORK) or out == WORK:
        raise ValueError("output must be a fresh child of the signed benchmark task root")
    for name in ("TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "MPLCONFIGDIR", "XDG_CACHE_HOME",
                 "NUMBA_CACHE_DIR", "JOBLIB_TEMP_FOLDER", "PYTHONUSERBASE", "PIP_CACHE_DIR"):
        if not Path(os.environ.get(name, "")).resolve().is_relative_to(WORK):
            raise RuntimeError(f"repo-local signed-experiment runtime required: {name}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or version("lightgbm") != "4.6.0":
        raise RuntimeError("explicit CPU and LightGBM4.6.0 required")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=out / "runtime.log", level=logging.INFO)
    inputs = {}

    def checked(path, digest):
        if sha(path) != digest:
            raise RuntimeError(f"input identity mismatch: {path}")
        inputs[Path(path).relative_to(ROOT).as_posix()] = digest
        return Path(path)

    inventory = json.loads(checked(OLD / "session-artifacts.json",
        "3a0ac1e3dcdc9777c4a2f75bf82a606a0b3b82f5cd66f34946dc09c3ce6dde1a").read_text())

    def old(name):
        return checked(OLD / name, inventory["files"][name]["sha256"])

    old_plan = json.loads(old("plan.json").read_text())
    capture = json.loads(checked(WORK / "baseline/manifest.json",
        "f51fb0e62b13a9eba6c86e55d8f3dd654ee3edf98264b981ae69d683c54ab129").read_text())
    code = {}
    for name, digest in capture["files_sha256"].items():
        checked(WORK / "baseline/sources" / name, digest)
        current = sha(ROOT / name)
        if current != digest and name != "pfc_shaping/lt/model/assembler.py":
            raise RuntimeError(f"unexpected baseline dependency change: {name}")
        code[name] = current
    for name in ("scripts/run_lt_signed_benchmark.py", "pfc_shaping/lt/signed_benchmark.py",
                 "pfc_shaping/lt/structural_readiness.py", "tests/test_signed_hourly_assembly.py",
                 "tests/test_signed_benchmark.py", "docs/model/LT-SIGNED-SHAPE-LOCAL-EXPERIMENT.md"):
        code[name] = sha(ROOT / name)
    actual_path = checked(SOURCE / "prepared-inputs/epex-ch.parquet",
        "dfda560ba9ae706704653805e31599231d6fd345b9a18f130b74f744de1f9c84")
    checked(SOURCE / "prepared-inputs/manifest.json",
        "36f27a705bdf56ff723d778d226ed3b5f4ae594f20e54b402cc3d6385088b3ef")
    checked(SOURCE / "curve-final/pfc-fmv-ch-15min.csv",
        "d35599fa0aaf51b2e89b85857f2cf4a814e624dbef51375cbf52ae59d3f16b02")
    old("comparative-summary.csv")
    old("selection.json")
    for name in inventory["files"]:
        if name.startswith("dependencies/lightgbm/"):
            old(name)
    for spec in old_plan["folds"]:
        for name in ("training-hydro.parquet", "solver.json", "monthly-curve.parquet",
                     "common-native-curve.parquet", "eex-surface.parquet", "run-complete.json",
                     f"{CORRECTED}.pkl"):
            old(f"{spec['id']}/{name}")
    plan = dict(schema="fmv-lt-signed-representation-local.v1", frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        candidates=CANDIDATES, parameters=PARAMETERS, folds=old_plan["folds"],
        source_policy="LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT",
        comparison_policy="PREVIOUSLY_EXPOSED_DEVELOPMENT_YEARS_NO_NEW_SELECTION",
        truth_end_exclusive="2026-09-01T00:00:00+02:00", estimator_fits_planned=12,
        seasonal_calculations_planned=12, authority=dict(AUTHORITIES),
        code_sha256=code, inputs_sha256=dict(inputs),
        versions={key: version(key) for key in ("numpy", "pandas", "scikit-learn", "lightgbm")})
    write_json(out / "plan.json", plan)
    print(json.dumps(dict(stage="plan_frozen", plan_sha256=sha(out / "plan.json"))), flush=True)
    actual_qh = pd.read_parquet(actual_path)["price_eur_mwh"]
    grouped = actual_qh.resample("h")
    if not grouped.count().eq(4).all() or not (grouped.max() - grouped.min()).eq(0).all():
        raise ValueError("CH transport must agree in every native hour")
    actual_hourly = grouped.mean()
    truth = actual_qh.loc[actual_qh.index < pd.Timestamp(plan["truth_end_exclusive"])]
    metrics, records, controls, year_scores = [], [], [], []
    for spec in plan["folds"]:
        origin, year = pd.Timestamp(spec["origin_utc"]), spec["id"]
        folder = out / year
        folder.mkdir()
        targets = closed_month_targets(actual_hourly, origin)
        targets.to_parquet(folder / "targets.parquet")
        hydro = pd.read_parquet(old(f"{year}/training-hydro.parquet"))
        if hydro.index.max() >= origin:
            raise ValueError("hydro training must precede origin")
        mapper = HydroAlignedShapeHourlyMLP()
        mapper._setup_hydro(hydro)
        times = targets.index
        cal = enrich_15min_index(times)
        local = times.tz_convert("Europe/Zurich")
        x = _encode_features(local.hour.to_numpy(), local.month.to_numpy(), local.dayofweek.to_numpy(),
            cal.type_jour.isin(["Ferie_CH", "Ferie_DE"]).to_numpy(), mapper._map_hydro_fill(times),
            np.zeros(len(times)))[:, :9]
        # Fit only the native weekday table on the same closed-month observations.
        training_grid = pd.date_range(times[0], times[-1] + pd.Timedelta(hours=1), freq="15min", inclusive="left")
        training = actual_qh.reindex(training_grid).to_frame("price_eur_mwh")
        training = training.join(enrich_15min_index(training_grid)[["type_jour"]])
        daily = training.price_eur_mwh.groupby(training.index.tz_convert("Europe/Zurich").normalize()).transform("mean")
        keep = daily.gt(5)
        weights = np.exp2(-(training.index.max() - training.index).total_seconds().to_numpy() / 86400 / mapper.halflife_days)
        mapper._fit_f_W(training.loc[keep], weights[keep], None)
        solver = json.loads(old(f"{year}/solver.json").read_text())
        saved = pd.read_parquet(old(f"{year}/common-native-curve.parquet"))
        grid = saved.index
        hourly_grid = grid[::4]
        xp = prediction_matrix(hourly_grid, origin, mapper)
        np.savez_compressed(folder / "features.npz", x=x, xp=xp,
            training_timestamp_ns=times.as_unit("ns").asi8, prediction_timestamp_ns=hourly_grid.as_unit("ns").asi8)
        shared = dict(base_prices=solver["base_prices"], quoted_keys=set(solver["quoted_keys"]),
            delivery_index=grid, reference_date=origin, country="CH")
        with old(f"{year}/{CORRECTED}.pkl").open("rb") as stream:
            incumbent = pickle.load(stream)
        native = make_assembler(incumbent).build(**shared)
        legacy_error = float((native.price_shape - saved.price_shape).abs().max())
        if legacy_error > 1e-10 or not native.index.equals(saved.index):
            raise RuntimeError("legacy path changed from D301")
        prior = center_signed_hourly_shape(saved.price_pre_final_projection.resample("h").mean())
        roundtrip = make_assembler(incumbent).build(**shared, signed_hourly_shape=prior)
        roundtrip_error = float((roundtrip.price_shape.resample("h").mean() - saved.price_shape.resample("h").mean()).abs().max())
        if roundtrip_error > 1e-9:
            raise RuntimeError("signed assembly round trip changed native hourly prices")
        roundtrip.to_parquet(folder / "roundtrip.parquet")
        controls.append(dict(origin=year, legacy_qh_max_error=legacy_error, signed_roundtrip_hourly_max_error=roundtrip_error))
        write_json(folder / "preparation.json", dict(origin=origin.isoformat(), rows=len(targets),
            first_training_hour=times[0].isoformat(), last_training_hour=times[-1].isoformat(),
            signed_negative_price_hours=int(targets.price_eur_mwh.lt(0).sum()),
            ratio_excluded_hours=int(targets.ratio_target.isna().sum()), native_f_W=mapper.f_W_,
            authority=dict(AUTHORITIES)))
        surface = pd.read_parquet(old(f"{year}/eex-surface.parquet"))
        monthly = pd.read_parquet(old(f"{year}/monthly-curve.parquet")).iloc[:, 0]
        common_population = None
        for candidate in ("d301-current-mlp",) + CANDIDATES:
            candidate_folder = folder / candidate
            candidate_folder.mkdir()
            started = time.perf_counter()
            record = dict(origin=year, role=spec["role"], candidate=candidate,
                estimator_fit=False, seasonal_calculation=False, authority=dict(AUTHORITIES))
            try:
                if candidate == "d301-current-mlp":
                    frame = native
                    record["fit_seconds"] = 0.
                else:
                    signed = candidate.startswith("signed")
                    label = "signed_target" if signed else "ratio_target"
                    target = targets[label].dropna()
                    mask = targets.index.isin(target.index)
                    if candidate.endswith("lightgbm"):
                        record["estimator_fit"] = True
                        model = fit_challenger(CHALLENGERS[-1], x[mask], target.to_numpy(), target.index, origin, PARAMETERS)
                        record["estimator_fit_completed"] = True
                        record["fit_seconds"] = time.perf_counter() - started
                        with (candidate_folder / "model.pkl").open("xb") as stream:
                            pickle.dump(model, stream, protocol=5)
                        raw = predict_challenger(CHALLENGERS[-1], model, xp)
                    else:
                        raw, counts = calendar_cell_reference(target, hourly_grid)
                        record.update(seasonal_calculation=True, reference_cell_counts=counts,
                                      fit_seconds=time.perf_counter() - started)
                    np.save(candidate_folder / "raw-hourly-prediction.npy", raw)
                    if signed:
                        shaped = center_signed_hourly_shape(pd.Series(raw, index=hourly_grid))
                        assembly = make_assembler(mapper)
                        frame = assembly.build(**shared, signed_hourly_shape=shaped)
                    else:
                        factors = postprocess(np.repeat(raw, 4), grid)
                        assembly = make_assembler(_InjectedHourlyFactors(mapper, grid, factors))
                        frame = assembly.build(**shared)
                    record["projection"] = assembly.final_product_projection_report_
                if not np.isfinite(frame.price_shape).all():
                    raise ValueError("nonfinite assembled prices")
                months = grid.tz_convert("Europe/Zurich").tz_localize(None).to_period("M")
                record["max_monthly_solver_error"] = float((frame.price_shape.groupby(months).mean() - monthly).abs().max())
                if record["max_monthly_solver_error"] > 1e-9:
                    raise ValueError("candidate rewrote solver monthly levels")
                frame.to_parquet(candidate_folder / "curve.parquet")
                hourly = frame.price_shape.resample("h").mean().to_frame("price_eur_mwh")
                hourly["ts_ch"] = hourly.index.tz_convert("Europe/Zurich")
                for field in ("year", "month", "quarter"):
                    hourly[field] = getattr(hourly.ts_ch.dt, field)
                gates = build_product_normalization_gates(hourly, surface,
                    forward_date=pd.Timestamp(surface.date.iloc[0]), price_column="price_eur_mwh",
                    hard_tolerance=1e-6, peak_country="CH")
                gates.to_csv(candidate_folder / "product-gates.csv", index=False)
                record["market_gate_counts"] = {str(k): int(v) for k, v in gates.status.value_counts().items()}
                if gates.status.eq("CRITICAL").any():
                    raise ValueError("critical market-consistency gate")
                for stage, column in (("final", "price_shape"), ("pre_projection", "price_pre_final_projection")):
                    scores, errors, coverage = score_curves(frame[[column]].rename(columns={column: candidate}), truth, origin)
                    scores["origin"], scores["role"], scores["stage"] = year, spec["role"], stage
                    scores.to_csv(candidate_folder / f"metrics-{stage}.csv", index=False)
                    errors.to_parquet(candidate_folder / f"errors-{stage}.parquet")
                    metrics.append(scores)
                    if common_population is None:
                        common_population = errors.index
                    if not common_population.equals(errors.index):
                        raise ValueError("candidate changed the common evaluation population")
                    if stage == "final":
                        record["overall"] = scores.loc[scores.segment.eq("ALL")].iloc[0].to_dict()
                        record["coverage"] = coverage
                        eligible = errors.index
                        record["negative_predicted_eligible_hours"] = int(hourly.price_eur_mwh.reindex(eligible).lt(0).sum())
                        for y in (1, 2, 3):
                            e = errors.iloc[:, 0].loc[eligible.tz_convert("Europe/Zurich").year == int(year) + y - 1]
                            if len(e):
                                year_scores.append(dict(origin=year, role=spec["role"], candidate=candidate,
                                    delivery_year=y, hours=len(e), mae=float(e.abs().mean()), rmse=float(np.sqrt(np.mean(e ** 2)))))
                record["status"] = "SCORED_LOCAL_DEVELOPMENT"
            except Exception as exc:
                record.update(status="FAILED_NOT_RANKABLE", error_type=type(exc).__name__, error=str(exc))
                (candidate_folder / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            record["elapsed_seconds"] = time.perf_counter() - started
            record["files_sha256"] = {p.name: sha(p) for p in candidate_folder.iterdir() if p.is_file()}
            write_json(candidate_folder / "result.json", record)
            records.append(record)
            print(json.dumps({k: record[k] for k in ("origin", "candidate", "status", "elapsed_seconds")}), flush=True)
    scored = pd.concat(metrics, ignore_index=True)
    statuses = pd.DataFrame([{k: r[k] for k in ("origin", "candidate", "status")} for r in records])
    scored = scored.merge(statuses.rename(columns={"status": "candidate_status"}), on=["origin", "candidate"], validate="many_to_one")
    scored.to_csv(out / "all-metrics.csv", index=False)
    pd.DataFrame(year_scores).to_csv(out / "by-delivery-year.csv", index=False)
    comparisons = scored.loc[scored.role.eq("assessment") & scored.candidate_status.eq("SCORED_LOCAL_DEVELOPMENT")].groupby(["candidate", "stage", "segment"], as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), p95=("p95", "mean"),
        supported_origins=("mae", "count"), origin_hours=("hours", "sum"))
    comparisons.to_csv(out / "comparative-summary.csv", index=False)
    for name, digest in inputs.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("input changed during execution")
    for name, digest in code.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("code changed during execution")
    write_json(out / "complete.json", dict(status="LOCAL_EXPERIMENT_EXECUTED_NO_PROMOTION", records=records,
        controls=controls, estimator_fits=sum(r["estimator_fit"] for r in records),
        seasonal_calculations=sum(r["seasonal_calculation"] for r in records),
        failures=[r for r in records if r["status"] != "SCORED_LOCAL_DEVELOPMENT"],
        completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat(), authority=dict(AUTHORITIES)))
    logging.shutdown()
    write_json(out / "manifest.json", {p.relative_to(out).as_posix(): sha(p)
        for p in out.rglob("*") if p.is_file()})
    print(comparisons.loc[comparisons.segment.eq("ALL")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
