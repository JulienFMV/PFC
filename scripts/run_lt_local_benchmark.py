"""Reproduce the authorized local CH hourly benchmark on retained D300 bytes.

Run with the repo-local Python module runtime and caches below build. Stages:
freeze (no fit), run (CPU fits/predictions/scoring). Existing receipts are never
overwritten. A changed implementation requires a new frozen experiment root.
"""

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
import warnings

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build/local-lt-benchmark-20260907"
SOURCE = ROOT / "build/local-pfc-source-preflight-20260907"
if Path.cwd() != ROOT or str(ROOT).lower() != r"c:\users\jbattaglia\pfc_lt":
    raise RuntimeError("local benchmark requires canonical workstation checkout")
for key in ("TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "MPLCONFIGDIR", "XDG_CACHE_HOME",
            "NUMBA_CACHE_DIR", "JOBLIB_TEMP_FOLDER", "PYTHONUSERBASE", "PIP_CACHE_DIR"):
    folder = WORK / "runtime" / key.lower()
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.append(str(WORK / "dependencies"))

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from importlib.metadata import version

from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
from pfc_shaping.calibration.monthly_forward_curve import product_periods
from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.databricks_eex_daily_snapshot import (
    select_latest_quote_surface, split_solver_quote_maps,
)
from pfc_shaping.lt.evaluation_curve_assembly import (
    _InjectedHourlyFactors, _build_from_template, _validate_common_assembly,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.local_benchmark import (
    AUTHORITIES, CHALLENGERS, CORRECTED, FEATURES, FROZEN, REFERENCE, SCHEMA,
    complete_training_days, fit_challenger, parameter_grid, postprocess,
    predict_challenger, prediction_matrix, score_curves, seasonal_reference, training_matrix,
)
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from pfc_shaping.pipeline.monthly_curve_authority import (
    delivery_months_from_prices, monthly_solver_settings, solve_monthly_level_authority,
)
from pfc_shaping.validation.product_normalization import build_product_normalization_gates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def emit(**values):
    print(json.dumps(dict(at_utc=pd.Timestamp.now(tz="UTC").isoformat(), **values)), flush=True)


def identities():
    source_files = [SOURCE / "prepared-inputs/manifest.json", SOURCE / "eex-replay/audit.json",
                    SOURCE / "eex-replay/eex-normalized-history.parquet"]
    source_files += [SOURCE / "prepared-inputs" / name for name in
                     ("epex-ch.parquet", "hydro.parquet")]
    # These bytes are part of the replay, not just the top-level script.
    code_files = [Path(__file__), ROOT / "docs/model/LT-LOCAL-CPU-BENCHMARK.md",
                  ROOT / "pfc_shaping/config.yaml"]
    for relative in ("pfc_shaping/lt", "pfc_shaping/calibration"):
        code_files += sorted((ROOT / relative).rglob("*.py"))
    code_files += [ROOT / name for name in (
        "pfc_shaping/pipeline/monthly_curve_authority.py",
        "pfc_shaping/data/calendar_ch.py", "pfc_shaping/data/databricks_eex_daily_snapshot.py",
        "pfc_shaping/validation/product_normalization.py",
    )]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
            for p in source_files + code_files}


def load_sources():
    manifest = json.loads((SOURCE / "prepared-inputs/manifest.json").read_text())
    for name in ("epex-ch.parquet", "hydro.parquet"):
        if sha(SOURCE / "prepared-inputs" / name) != manifest["files"][name]:
            raise RuntimeError(f"D300 prepared source changed: {name}")
    audit = json.loads((SOURCE / "eex-replay/audit.json").read_text())
    if sha(SOURCE / "eex-replay/eex-normalized-history.parquet") != audit["output_sha256"]:
        raise RuntimeError("D300 EEX source changed")
    return (pd.read_parquet(SOURCE / "prepared-inputs/epex-ch.parquet"),
            pd.read_parquet(SOURCE / "prepared-inputs/hydro.parquet")[["fill_pct"]],
            pd.read_parquet(SOURCE / "eex-replay/eex-normalized-history.parquet"))


def freeze():
    if (WORK / "plan.json").exists():
        raise RuntimeError("experiment already frozen; do not overwrite")
    prices, hydro, history = load_sources()
    folds = [dict(id=str(year), role="tuning" if year < 2023 else "assessment",
                  origin_utc=f"{year-1}-12-31T12:00:00+00:00", months=12 if year < 2023 else 36)
             for year in range(2021, 2027)]
    plan = dict(schema=SCHEMA, frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                sources_and_code_sha256=identities(), folds=folds,
                candidate_ids=[REFERENCE, CORRECTED, FROZEN, *CHALLENGERS],
                grids={c: parameter_grid(c) for c in CHALLENGERS},
                feature_names=list(FEATURES), protocol_v6_sha256=default_evaluation_protocol().semantic_sha256(),
                authority=dict(AUTHORITIES), real_execution_user_authorization="EXPLICIT_LOCAL_CPU_BENCHMARK",
                settings=monthly_solver_settings(yaml.safe_load((ROOT / "pfc_shaping/config.yaml").read_bytes())),
                software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                              scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=version("lightgbm")),
                lightgbm_install_sha256=sha(WORK / "lightgbm-install.json"),
                truth_end_exclusive="2026-09-01T00:00:00+02:00", source_policy="LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT")
    plan["settings"]["eex_history_path"] = str(SOURCE / "eex-replay/eex-normalized-history.parquet")
    write_json(WORK / "plan.json", plan)
    emit(stage="plan_frozen", plan_sha256=sha(WORK / "plan.json"), folds=len(folds))
    for spec in folds:
        origin = pd.Timestamp(spec["origin_utc"])
        folder = WORK / spec["id"]
        folder.mkdir(exist_ok=False)
        train = complete_training_days(prices, origin)
        h = hydro.loc[hydro.index < origin].copy()
        x, y, timestamps, mapper = training_matrix(train, h, origin)
        train.to_parquet(folder / "training-prices.parquet")
        h.to_parquet(folder / "training-hydro.parquet")
        np.savez_compressed(folder / "training-matrix.npz", x=x, y=y,
                            timestamp_ns=timestamps.as_unit("ns").asi8)
        cutoff = origin.tz_convert("Europe/Zurich").tz_localize(None).normalize()
        old_history = history.loc[history["date"] < cutoff].copy()
        surface = select_latest_quote_surface(old_history)
        first = pd.Period(f"{spec['id']}-01", freq="M")
        last = first + spec["months"] - 1
        keep = surface["product"].map(lambda p: product_periods(str(p))[0] >= first
                                      and product_periods(str(p))[-1] <= last)
        surface = surface.loc[keep].copy()
        quotes = split_solver_quote_maps(surface)
        own = dict(quotes["BASE"])
        own.update({k + "-Peak": v for k, v in quotes["PEAK"].items()})
        months = delivery_months_from_prices(own)
        if months[0] != first or months[-1] != last:
            raise RuntimeError(f"EEX surface cannot cover frozen delivery window {spec['id']}")
        old_history.to_parquet(folder / "eex-history.parquet")
        surface.to_parquet(folder / "eex-surface.parquet")
        authority = solve_monthly_level_authority(
            market="CH", delivery_months=months, own_base_prices=own, all_market_base_prices={},
            eex_history=old_history, run_timestamp=origin, settings=plan["settings"], timezone="Europe/Zurich",
            source_hashes={"retained_prd_eex": sha(SOURCE / "eex-replay/eex-normalized-history.parquet"),
                           "quotation_cutoff_frame": sha(folder / "eex-history.parquet")},
            original_forward_prices=own, allow_unverified_inputs=True,
        )
        index = pd.date_range(first.start_time, (last + 1).start_time, freq="15min",
                              inclusive="left", tz="Europe/Zurich").tz_convert("UTC")
        xp = prediction_matrix(index, origin, mapper)
        np.savez_compressed(folder / "prediction-matrix.npz", x=xp, timestamp_ns=index.as_unit("ns").asi8)
        authority.result.monthly_curve.to_frame("monthly_base").to_parquet(folder / "monthly-curve.parquet")
        write_json(folder / "solver.json", dict(manifest=authority.manifest,
                   base_prices=authority.assembler_base_prices, quoted_keys=sorted(authority.quoted_keys)))
        truth = prices["price_eur_mwh"].loc[prices.index < pd.Timestamp(plan["truth_end_exclusive"])]
        # Coverage only: zero predictions do not fit or rank a model.
        _, _, coverage = score_curves(pd.DataFrame({"coverage_probe": 0.0}, index=index), truth, origin)
        day_means = train.price_eur_mwh.groupby(train.index.tz_convert("Europe/Zurich").normalize()).transform("mean")
        receipt = dict(**spec, training_qh=len(train), training_hours=len(y),
                       training_end_utc=train.index[-1].isoformat(), hydro_end_utc=h.index[-1].isoformat(),
                       excluded_incomplete_training_qh=int((prices.index < origin).sum()) - len(train),
                       training_native_filter_excluded_qh=int(day_means.le(5).sum()),
                       native_repeated_hour_merged_qh=int(day_means.gt(5).sum())-4*len(y),
                       quote_date=str(surface["date"].iloc[0].date()), quote_count=len(surface),
                       prediction_qh=len(index), coverage=coverage,
                       files={p.name: sha(p) for p in folder.iterdir() if p.is_file()},
                       authority=dict(AUTHORITIES), fit_performed=False,
                       score_probe="COVERAGE_ONLY_ZERO_PLACEHOLDER_NEVER_CANDIDATE_SCORE")
        write_json(folder / "inputs.json", receipt)
        emit(stage="origin_prepared", fold=spec["id"], training_hours=len(y),
             supported_months=sum(r["eligible"] for r in coverage), quote_count=len(surface))
    write_json(WORK / "freeze-complete.json", dict(plan_sha256=sha(WORK / "plan.json"),
               folds={s["id"]: sha(WORK / s["id"] / "inputs.json") for s in folds}))


def checked_plan():
    plan = json.loads((WORK / "plan.json").read_text())
    receipt = json.loads((WORK / "freeze-complete.json").read_text())
    if receipt["plan_sha256"] != sha(WORK / "plan.json") or plan["sources_and_code_sha256"] != identities():
        raise RuntimeError("frozen plan/source/code mismatch")
    for spec in plan["folds"]:
        folder = WORK / spec["id"]
        if sha(folder / "inputs.json") != receipt["folds"][spec["id"]]:
            raise RuntimeError("frozen origin receipt mismatch")
        for name, digest in json.loads((folder / "inputs.json").read_text())["files"].items():
            if sha(folder / name) != digest:
                raise RuntimeError(f"frozen input mismatch: {folder.name}/{name}")
    return plan


def fit_native(folder, candidate, train, hydro, cal):
    path = folder / f"{candidate}-fit.json"
    model_path = folder / f"{candidate}.pkl"
    if path.exists():
        receipt = json.loads(path.read_text())
        if sha(model_path) != receipt["model_sha256"]:
            raise RuntimeError("cached native model changed")
        with model_path.open("rb") as stream:
            return pickle.load(stream), receipt
    start = time.perf_counter()
    cls = HydroAlignedShapeHourlyMLP if candidate == CORRECTED else ShapeHourlyMLP
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = cls().fit(train, cal, hydro_df=hydro, outages_df=None)
    with model_path.open("xb") as stream:
        pickle.dump(model, stream, protocol=5)
    receipt = dict(candidate=candidate, fit_seconds=time.perf_counter()-start,
                   native_iterations=int(model.mlp_.n_iter_), warnings=[str(w.message) for w in caught],
                   model_sha256=sha(model_path), status="FITTED_NATIVE_LOCAL", authority=dict(AUTHORITIES))
    write_json(path, receipt)
    emit(stage="native_fit_completed", fold=folder.name, **receipt)
    return model, receipt


def run_fold(spec, selected):
    folder = WORK / spec["id"]
    if (folder / "run-complete.json").exists():
        return
    origin = pd.Timestamp(spec["origin_utc"])
    train = pd.read_parquet(folder / "training-prices.parquet")
    hydro = pd.read_parquet(folder / "training-hydro.parquet")
    matrix = np.load(folder / "training-matrix.npz")
    x, y = matrix["x"], matrix["y"]
    times = pd.to_datetime(matrix["timestamp_ns"], utc=True)
    prediction = np.load(folder / "prediction-matrix.npz")
    xp, index = prediction["x"], pd.to_datetime(prediction["timestamp_ns"], utc=True)
    cal = enrich_15min_index(train.index)
    corrected, corrected_fit = fit_native(folder, CORRECTED, train, hydro, cal)
    frozen, frozen_fit = fit_native(folder, FROZEN, train, hydro, cal)
    template = PFCAssembler(
        corrected, ShapeIntraday(), uncertainty=None, water_value=None, cascader=None,
        calibrator=ArbitrageFreeCalibrator(smoothness_weight=1., tol=.01),
        calibration_fallback_to_raw=False, peak_source_policy="same_first",
        monthly_level_authority="solver", skip_legacy_level_cascade=True,
        skip_legacy_base_smoothing=True, monthly_constraint_tolerance=1e-9,
        allow_negative_prices=False,
    )
    solver = json.loads((folder / "solver.json").read_text())
    shared = dict(base_prices=solver["base_prices"], quoted_keys=set(solver["quoted_keys"]),
                  delivery_index=index, entso_forecast=None, hydro_forecast=None,
                  outages_forecast=None, reference_date=origin, country="CH")
    native_frame = _build_from_template(template, corrected, shared)
    native_frame.to_parquet(folder / "common-native-curve.parquet")
    if not np.all(native_frame["f_Q"] == 1) or not np.all(native_frame["delta_wv"] == 0):
        raise RuntimeError("hourly experiment unexpectedly used an intraday/water layer")
    surface = pd.read_parquet(folder / "eex-surface.parquet")
    actual = pd.read_parquet(SOURCE / "prepared-inputs/epex-ch.parquet")["price_eur_mwh"]
    actual = actual.loc[actual.index < pd.Timestamp("2026-09-01", tz="Europe/Zurich")]
    inventory = [(REFERENCE, {}), (CORRECTED, {}), (FROZEN, {})]
    for candidate in CHALLENGERS:
        configs = parameter_grid(candidate) if spec["role"] == "tuning" else [selected[candidate]]
        inventory += [(candidate, config) for config in configs if config is not None]
    results = []
    for candidate, config in inventory:
        key = candidate + "-" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
        out = folder / key
        if (out / "result.json").exists():
            result = json.loads((out / "result.json").read_text())
            for name, digest in result.get("files", {}).items():
                if sha(out / name) != digest:
                    raise RuntimeError("cached result changed")
            results.append(result)
            continue
        out.mkdir(exist_ok=False)
        started = time.perf_counter()
        result = dict(candidate=candidate, parameters=config, fold=spec["id"], role=spec["role"],
                      authority=dict(AUTHORITIES), key=key)
        emit(stage="candidate_started", fold=spec["id"], candidate=candidate, parameters=config)
        try:
            if candidate == CORRECTED:
                frame = native_frame.copy()
                result["fit_seconds"] = corrected_fit["fit_seconds"]
            elif candidate == FROZEN:
                frame = _build_from_template(template, frozen, shared)
                result["fit_seconds"] = frozen_fit["fit_seconds"]
            else:
                if candidate == REFERENCE:
                    factors, fallback = seasonal_reference(y, times, index)
                    result["reference_cell_counts"] = fallback
                else:
                    model = fit_challenger(candidate, x, y, times, origin, config)
                    with (out / "model.pkl").open("xb") as stream:
                        pickle.dump(model, stream, protocol=5)
                    result["fit_seconds"] = time.perf_counter()-started
                    factors = postprocess(predict_challenger(candidate, model, xp), index)
                result.setdefault("fit_seconds", time.perf_counter()-started)
                injected = _InjectedHourlyFactors(corrected, index, factors)
                frame = _build_from_template(template, injected, shared)
            _validate_common_assembly({candidate: frame}, native_frame, index)
            frame[["price_shape", "f_H"]].to_parquet(out / "prediction.parquet")
            # Same real quote surface for every model; no local waiver of conflicts.
            hourly = frame["price_shape"].resample("h").mean().to_frame("price_eur_mwh")
            hourly["ts_ch"] = hourly.index.tz_convert("Europe/Zurich")
            for field in ("year", "month", "quarter"):
                hourly[field] = getattr(hourly["ts_ch"].dt, field)
            gates = build_product_normalization_gates(hourly, surface,
                        forward_date=pd.Timestamp(surface["date"].iloc[0]),
                        price_column="price_eur_mwh", hard_tolerance=1e-6, peak_country="CH")
            gates.to_csv(out / "product-gates.csv", index=False)
            result["market_gate_counts"] = {str(k): int(v) for k, v in gates["status"].value_counts().items()}
            if gates["status"].eq("CRITICAL").any():
                raise RuntimeError("critical market-consistency gate; candidate not rankable")
            scores, errors, coverage = score_curves(frame[["price_shape"]].rename(columns={"price_shape": candidate}),
                                                     actual, origin)
            scores.to_csv(out / "metrics.csv", index=False)
            errors.to_parquet(out / "errors-hourly.parquet")
            result["overall"] = scores.loc[scores.segment == "ALL"].iloc[0].to_dict()
            result["status"] = "SCORED_LOCAL_RETROSPECTIVE"
            result["negative_predicted_qh"] = int(frame.price_shape.lt(0).sum())
            result["coverage"] = coverage
        except Exception as exc:
            result.update(status="FAILED_NOT_RANKABLE", error_type=type(exc).__name__, error=str(exc))
            (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result["elapsed_seconds"] = time.perf_counter()-started
        result["files"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_json(out / "result.json", result)
        results.append(result)
        emit(stage="candidate_completed", fold=spec["id"], candidate=candidate,
             parameters=config, status=result["status"], seconds=result["elapsed_seconds"],
             mae=result.get("overall", {}).get("mae"), error=result.get("error"))
    write_json(folder / "run-complete.json", dict(results=results, authority=dict(AUTHORITIES),
                                                 completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat()))


def select_from_tuning(plan):
    rows = []
    selected = {}
    for candidate in CHALLENGERS:
        for config in parameter_grid(candidate):
            key = json.dumps(config, sort_keys=True)
            records = []
            for spec in plan["folds"][:2]:
                fold = json.loads((WORK / spec["id"] / "run-complete.json").read_text())
                records += [r for r in fold["results"] if r["candidate"] == candidate and r["parameters"] == config]
            ok = len(records) == 2 and all(r["status"] == "SCORED_LOCAL_RETROSPECTIVE" for r in records)
            rows.append(dict(candidate=candidate, parameter_key=key, eligible=ok,
                             mean_origin_mae=float(np.mean([r["overall"]["mae"] for r in records])) if ok else None))
        eligible = sorted((r for r in rows if r["candidate"] == candidate and r["eligible"]),
                          key=lambda r: (r["mean_origin_mae"], r["parameter_key"]))
        selected[candidate] = json.loads(eligible[0]["parameter_key"]) if eligible else None
    result = dict(selected=selected, tuning_grid_results=rows, rule="MEAN_ORIGIN_MAE_THEN_CANONICAL_PARAMETER_JSON",
                  selected_at_utc=pd.Timestamp.now(tz="UTC").isoformat(), assessment_used=False,
                  local_tuning_performed=True, authority=dict(AUTHORITIES))
    write_json(WORK / "selection.json", result)
    return selected


def run():
    plan = checked_plan()
    logging.basicConfig(filename=WORK / "execution.log", level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for spec in plan["folds"][:2]:
        run_fold(spec, None)
    if (WORK / "selection.json").exists():
        selected = json.loads((WORK / "selection.json").read_text())["selected"]
    else:
        selected = select_from_tuning(plan)
    emit(stage="tuning_locked_before_assessment", selected=selected)
    for spec in plan["folds"][2:]:
        run_fold(spec, selected)
    emit(stage="benchmark_execution_completed", authority=AUTHORITIES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else run()
