"""D307 fixed local CH hourly weighting benchmark; no operational authority."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from pfc_shaping.lt.local_benchmark import AUTHORITIES, score_curves
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference, closed_month_targets
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from scripts.run_lt_signed_composition import ROOT, OLD, SOURCE, SIGNED, assembler, save_curve, sha, write_json

WORK = ROOT / "build/lt-hourly-recency-20260908"
D306 = ROOT / "build/lt-price-conditioned-20260907/run-v1"
D305 = ROOT / "build/lt-signed-composition-20260907/run-v4"
CANDIDATES = {"signed-equal": None, "signed-hl365": 365.25, "signed-hl730": 730.5}


def diagnostic_frame(prediction, actual):
    if not prediction.index.equals(actual.index) or not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("diagnostics require aligned finite hourly truth and prediction")
    months = actual.index.tz_convert("Europe/Zurich").strftime("%Y-%m")
    predicted_mean = prediction.groupby(months).transform("mean")
    actual_mean = actual.groupby(months).transform("mean")
    contiguous = pd.Series(actual.index, index=actual.index).diff().eq(pd.Timedelta(hours=1))
    within = pd.Series(months, index=actual.index).eq(pd.Series(months, index=actual.index).shift()) & contiguous
    return pd.DataFrame(dict(actual=actual, prediction=prediction,
        full_error=prediction-actual, shape_error=prediction-predicted_mean-actual+actual_mean,
        level_error=predicted_mean-actual_mean,
        truth_ramp=actual.diff().where(within), ramp_error=(prediction.diff()-actual.diff()).where(within)))


def masks(frame, origin, thresholds):
    local = frame.index.tz_convert("Europe/Zurich")
    cal = enrich_15min_index(frame.index)
    leads = local.year*12 + local.month - (origin.tz_convert("Europe/Zurich").year*12 + origin.tz_convert("Europe/Zurich").month)
    peak = (local.dayofweek<5) & (local.hour>=8) & (local.hour<20)
    result = {"ALL": np.ones(len(frame),bool), "NEGATIVE_TRUTH": frame.actual.to_numpy()<0,
        "NEAR_ZERO_TRUTH": frame.actual.abs().to_numpy()<=5, "POSITIVE_TRUTH": frame.actual.to_numpy()>5,
        "HIGH_PRICE": frame.actual.to_numpy()>thresholds["price_p95"],
        "LARGE_RAMP": frame.truth_ramp.abs().to_numpy()>thresholds["ramp_p95"],
        "PEAK_CLOCK": peak, "OFFPEAK_CLOCK": ~peak,
        "WEEKEND_OR_HOLIDAY": cal.type_jour.ne("Ouvrable").to_numpy(),
        "COMMON_FIRST8": (leads>=1)&(leads<=8)}
    for a,b in [(1,6),(7,12),(13,24),(25,36)]:
        result[f"M{a:02d}_M{b:02d}"]=(leads>=a)&(leads<=b)
    for season in ["Hiver","Printemps","Ete","Automne"]:
        result[season]=cal.saison.eq(season).to_numpy()
    for year in sorted(set(local.year)):
        result[f"YEAR_{year}"]=local.year==year
    return result


def stats(error):
    error = error.dropna()
    return dict(hours=len(error), mae=float(error.abs().mean()) if len(error) else None,
        rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        bias=float(error.mean()) if len(error) else None,
        p95=float(error.abs().quantile(.95)) if len(error) else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    out=parser.parse_args().output.resolve()
    if Path.cwd()!=ROOT or ROOT!=Path(r"C:\Users\jbattaglia\PFC_LT") or not out.is_relative_to(WORK) or out==WORK:
        raise ValueError("fresh canonical task output required")
    for name in ("TEMP","TMP","APPDATA","LOCALAPPDATA","MPLCONFIGDIR","XDG_CACHE_HOME","NUMBA_CACHE_DIR","JOBLIB_TEMP_FOLDER","PYTHONUSERBASE","PIP_CACHE_DIR"):
        if not Path(os.environ.get(name,"")).resolve().is_relative_to(WORK):
            raise ValueError(f"task-local runtime required: {name}")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise ValueError("CPU required")
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(filename=out/"runtime.log",level=logging.INFO)
    if sha(D306/"manifest.json")!="87ddf8e4f38fa78558d8877f7494e555a92270b99770c5cc55b88abb5d02111a":
        raise ValueError("D306 manifest mismatch")
    previous=json.loads((D306/"plan.json").read_text())
    pins=dict(previous["inputs_sha256"])
    pins[(D306/"manifest.json").relative_to(ROOT).as_posix()]=sha(D306/"manifest.json")
    for name,digest in json.loads((D306/"manifest.json").read_text()).items():
        pins[(D306/name).relative_to(ROOT).as_posix()]=digest
    code=dict(previous["code_sha256"])
    changed="pfc_shaping/lt/signed_benchmark.py"
    if sha(WORK/"baseline/signed_benchmark.py")!=code[changed]:
        raise ValueError("pre-edit reference mismatch")
    for name,digest in {**pins,**code}.items():
        if name!=changed and sha(ROOT/name)!=digest:
            raise ValueError(f"bound source/input mismatch: {name}")
    for name in (changed,"scripts/run_lt_hourly_recency.py","tests/test_hourly_recency.py",
        "docs/model/LT-HOURLY-RECENCY-LOCAL-EXPERIMENT.md","docs/data/CH-DAY-AHEAD-RESOLUTION-20260908.md"):
        code[name]=sha(ROOT/name)
    for name in ("scripts/verify_lt_price_conditioned_intraday.py",):
        code[name]=sha(ROOT/name)
    for name in ("source-review.json",):
        path=WORK/"web-sources"/name
        pins[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,digest in code.items():
        target=out/"source-snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    folds=json.loads((D305/"plan.json").read_text())["folds"]
    plan=dict(schema="fmv-hourly-recency.v1",frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        candidates=CANDIDATES,folds=folds,inputs_sha256=pins,code_sha256=code,authority=dict(AUTHORITIES),
        valuation="RETAINED_20260907_NOT_REPRICED",statistical_estimator_fits=0)
    write_json(out/"plan.json",plan)
    print(json.dumps(dict(stage="plan_frozen",sha256=sha(out/"plan.json"))),flush=True)
    truth=pd.read_parquet(SOURCE/"prepared-inputs/epex-ch.parquet").price_eur_mwh
    grouped=truth.resample("h")
    if not grouped.count().eq(4).all() or not (grouped.max()-grouped.min()).eq(0).all():
        raise ValueError("CH truth is no longer repeated hourly")
    truth=truth.loc[truth.index<pd.Timestamp("2026-09-01",tz="Europe/Zurich")]
    hourly_model=HydroAlignedShapeHourlyMLP.load(SOURCE/"fitted-models/hourly.pkl")
    reference=pd.Timestamp(json.loads((SOURCE/"curve-final/manifest.json").read_text())["valuation_at_utc"])
    scores, diagnostics, receipts, controls, monthly_records=[],[],[],[],[]
    for spec in folds+[dict(id="current",role="descriptive",origin_utc=reference.isoformat())]:
        label,origin=spec["id"],pd.Timestamp(spec["origin_utc"])
        current=label=="current"
        folder=out/label
        folder.mkdir()
        targets=closed_month_targets(truth.resample("h").mean(),origin)
        if not current:
            pd.testing.assert_frame_equal(targets,pd.read_parquet(SIGNED/f"{label}/targets.parquet"),check_freq=False)
        targets.to_parquet(folder/"targets.parquet")
        train_ramps=diagnostic_frame(targets.price_eur_mwh,targets.price_eur_mwh).truth_ramp
        thresholds=dict(price_p95=float(targets.price_eur_mwh.quantile(.95)),ramp_p95=float(train_ramps.abs().quantile(.95)))
        write_json(folder/"thresholds.json",thresholds)
        baseline=pd.read_parquet(D305/("current/signed/curve.parquet" if current else f"{label}/signed/curve.parquet"))
        solver=json.loads((SOURCE/"monthly-solver/result.json" if current else OLD/f"{label}/solver.json").read_text())
        levels=solver["assembler_base_prices"] if current else solver["base_prices"]
        surface=(select_latest_quote_surface(pd.read_parquet(SOURCE/"eex-replay/eex-normalized-history.parquet")) if current else pd.read_parquet(OLD/f"{label}/eex-surface.parquet"))
        grid=baseline.index
        population=None
        for candidate,halflife in {"mlp-control":None,**CANDIDATES}.items():
            dest=folder/candidate
            if candidate=="mlp-control":
                frame=pd.read_parquet(D305/("current/mlp-current/curve.parquet" if current else f"{label}/mlp/curve.parquet"))
                dest.mkdir()
                frame.to_parquet(dest/"curve.parquet")
            else:
                age=(targets.index[-1]-targets.index).total_seconds().to_numpy()/86400
                weights=pd.Series(np.ones(len(targets)) if halflife is None else np.exp2(-age/halflife),index=targets.index)
                raw,counts=calendar_cell_reference(targets.signed_target,grid[::4],sample_weight=None if halflife is None else weights)
                assembly=assembler(hourly_model)
                frame=assembly.build(base_prices=levels,quoted_keys=set(solver["quoted_keys"]),delivery_index=grid,
                    reference_date=origin,country="CH",signed_hourly_shape=center_signed_hourly_shape(pd.Series(raw,index=grid[::4])))
                receipt=save_curve(dest,frame,levels,surface,assembly)
                receipts.append(dict(origin=label,candidate=candidate,**receipt))
                pd.DataFrame({"raw":raw},index=grid[::4]).to_parquet(dest/"raw.parquet")
                weights.to_frame("weight").to_parquet(dest/"weights.parquet")
                support=enrich_15min_index(targets.index)
                support["w"],support["w2"]=weights.to_numpy(),weights.to_numpy()**2
                tables=[]
                for keys in [["saison","type_jour","heure_hce"],["saison","heure_hce"],["heure_hce"]]:
                    table=support.groupby(keys).agg(rows=("w","size"),sum_weight=("w","sum"),sum_squared=("w2","sum"))
                    table["effective_hours"]=table.sum_weight**2/table.sum_squared
                    table["backoff"]="/".join(keys)
                    tables.append(table.reset_index())
                pd.concat(tables).to_csv(dest/"support.csv",index=False)
                write_json(dest/"fit.json",dict(halflife_days=halflife,rows=len(targets),counts=counts,
                    last_training=targets.index[-1].isoformat(),origin=origin.isoformat(),
                    global_effective_hours=float(weights.sum()**2/(weights**2).sum())))
                if halflife is None:
                    drift=float((frame.price_shape-baseline.price_shape).abs().max())
                    if drift>1e-9:
                        raise ValueError("D304 signed reference changed")
                    controls.append(dict(origin=label,max_error=drift))
                    if not current:
                        np.testing.assert_allclose(raw,np.load(SIGNED/f"{label}/signed-seasonal/raw-hourly-prediction.npy"),atol=1e-12,rtol=0)
            if current:
                main=frame.loc[grid<pd.Timestamp("2030-01-01",tz="Europe/Zurich")]
                for resolution,series in [("1h",main.price_shape.resample("h").mean()),("15min",main.price_shape)]:
                    pd.DataFrame({"timestamp_utc":series.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "timestamp_ch":series.index.tz_convert("Europe/Zurich").map(lambda t:t.isoformat()),
                        "price_eur_mwh":series.to_numpy()}).to_csv(dest/f"pfc-fmv-ch-{resolution}.csv",sep=";",index=False,float_format="%.10f")
                continue
            for stage,column in [("final","price_shape"),("pre_projection","price_pre_final_projection")]:
                metrics,errors,coverage=score_curves(frame[[column]].rename(columns={column:candidate}),truth,origin)
                if population is None: population=errors.index
                if not errors.index.equals(population): raise ValueError("candidate populations differ")
                errors.to_parquet(dest/f"errors-{stage}.parquet")
                metrics["origin"],metrics["role"],metrics["stage"]=label,spec["role"],stage
                scores.append(metrics)
                h=frame[column].resample("h").mean().reindex(population)
                actual=truth.resample("h").mean().reindex(population)
                diagnostic=diagnostic_frame(h,actual)
                np.testing.assert_allclose(diagnostic.shape_error,errors[candidate],atol=1e-10,rtol=0)
                diagnostic.to_parquet(dest/f"diagnostics-{stage}.parquet")
                for segment,mask in masks(diagnostic,origin,thresholds).items():
                    for error_type in ["shape_error","full_error","level_error","ramp_error"]:
                        diagnostics.append(dict(origin=label,role=spec["role"],candidate=candidate,stage=stage,
                            segment=segment,error_type=error_type,**stats(diagnostic.loc[mask,error_type])))
                month=population.tz_convert("Europe/Zurich").strftime("%Y-%m")
                for m,part in diagnostic.groupby(month):
                    monthly_records.append(dict(origin=label,candidate=candidate,stage=stage,month=m,hours=len(part),
                        predicted_mean=float(part.prediction.mean()),actual_mean=float(part.actual.mean()),
                        shape_mse=float(np.mean(part.shape_error**2)),level_mse=float(np.mean(part.level_error**2)),full_mse=float(np.mean(part.full_error**2))))
        print(json.dumps(dict(stage="origin_complete",origin=label)),flush=True)
    pd.concat(scores).to_csv(out/"native-metrics.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"diagnostic-metrics.csv",index=False)
    pd.DataFrame(monthly_records).to_csv(out/"monthly-decomposition.csv",index=False)
    for name,digest in {**pins,**code}.items():
        if sha(ROOT/name)!=digest: raise ValueError(f"bound bytes changed: {name}")
    write_json(out/"complete.json",dict(status="COMPLETE_LOCAL_NO_ADOPTION",controls=controls,receipts=receipts,
        seasonal_calculations=21,statistical_estimator_fits=0,authority=dict(AUTHORITIES),completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat()))
    logging.shutdown()
    write_json(out/"manifest.json",{p.relative_to(out).as_posix():sha(p) for p in out.rglob("*") if p.is_file()})


if __name__=="__main__":
    main()
