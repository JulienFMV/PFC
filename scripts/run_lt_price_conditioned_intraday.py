"""Frozen D306 CPU experiment using existing intraday estimators and assembler."""
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
from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference
from pfc_shaping.lt.signed_intraday import intraday_cell_reference
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, SIGNED, assembler, save_curve, sha, write_json

WORK = ROOT / "build/lt-price-conditioned-20260907"
PREVIOUS = ROOT / "build/lt-signed-composition-20260907/run-v4"
CANDIDATES = ("flat", "native", "native-regularized", "signed-additive")


def segments(parent, predicted, index):
    season = enrich_15min_index(index, country="DE").saison
    return {"ALL": np.ones(len(index), bool), "NEGATIVE_PARENT": parent < 0,
        "NEAR_ZERO_PARENT": np.abs(parent) <= 5, "POSITIVE_PARENT": parent > 5,
        "NEGATIVE_FORECAST": predicted < 0, "NEAR_ZERO_FORECAST": np.abs(predicted) <= 5,
        **{f"SEASON_{s}": season.to_numpy() == s for s in ("Hiver", "Printemps", "Ete", "Automne")}}


def runtime_guard():
    if Path.cwd() != ROOT or ROOT != Path(r"C:\Users\jbattaglia\PFC_LT"):
        raise RuntimeError("canonical workspace required")
    for name in ("TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "MPLCONFIGDIR", "XDG_CACHE_HOME",
                 "NUMBA_CACHE_DIR", "JOBLIB_TEMP_FOLDER", "PYTHONUSERBASE", "PIP_CACHE_DIR"):
        if not Path(os.environ.get(name, "")).resolve().is_relative_to(WORK):
            raise RuntimeError(f"task-local runtime required: {name}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_guard()
    out = args.output.resolve()
    if not out.is_relative_to(WORK) or out == WORK:
        raise ValueError("fresh task output required")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=out / "runtime.log", level=logging.INFO)
    prior_plan = json.loads((PREVIOUS / "plan.json").read_text())
    pins = dict(prior_plan["inputs_sha256"])
    for base, digest in ((PREVIOUS, "72509923aab50948116544ec3d3f85c6bcce4366896d938174899df068b87df2"),
                         (SIGNED, "446c940cc1a62845bbde664bdc61509078efca802467ccfd7451982031f134af")):
        if sha(base / "manifest.json") != digest:
            raise ValueError("prior manifest mismatch")
        pins[(base / "manifest.json").relative_to(ROOT).as_posix()] = digest
        for name, value in json.loads((base / "manifest.json").read_text()).items():
            pins[(base / name).relative_to(ROOT).as_posix()] = value
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"prior artifact mismatch: {name}")
    code = dict(prior_plan["code_sha256"])
    changed = "pfc_shaping/lt/model/shape_intraday.py"
    if sha(WORK / "baseline/shape_intraday.py") != code[changed]:
        raise ValueError("baseline component mismatch")
    for name, digest in code.items():
        if name != changed and sha(ROOT / name) != digest:
            raise ValueError(f"unexpected source change: {name}")
    for name in (changed, "scripts/run_lt_price_conditioned_intraday.py",
                 "tests/test_price_conditioned_intraday.py",
                 "docs/model/LT-PRICE-CONDITIONED-INTRADAY-EXPERIMENT.md"):
        code[name] = sha(ROOT / name)
    snapshot = out / "source-snapshot"
    for name, digest in code.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    plan = dict(schema="fmv-price-conditioned-intraday.v1", frozen_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        months=prior_plan["de_months"], candidates=CANDIDATES, inputs_sha256=pins,
        code_sha256=code, authority=dict(AUTHORITIES), protocol="docs/model/LT-PRICE-CONDITIONED-INTRADAY-EXPERIMENT.md")
    write_json(out / "plan.json", plan)
    print(json.dumps(dict(stage="plan_frozen", sha256=sha(out / "plan.json"))), flush=True)
    de = pd.read_parquet(SOURCE / "prepared-inputs/epex-de.parquet")
    metrics, controls = [], []
    for i, month in enumerate(plan["months"]):
        origin = pd.Timestamp(month + "-01", tz="Europe/Berlin")
        train = de.loc[de.index < origin]
        models = {c: ShapeIntraday.load(PREVIOUS / f"de-{month}/{c}.parquet") for c in CANDIDATES[1:3]}
        fit = json.loads((PREVIOUS / f"de-{month}/fit.json").read_text())
        if pd.Timestamp(fit["last_training_timestamp"]) >= origin or len(train) != fit["training_rows"]:
            raise ValueError("saved fit cutoff mismatch")
        for j, delivery in enumerate(plan["months"][i:], start=i):
            start = pd.Timestamp(delivery + "-01", tz="Europe/Berlin")
            end = (pd.Period(delivery, freq="M") + 1).start_time.tz_localize("Europe/Berlin")
            test = de.loc[(de.index >= start) & (de.index < end)].price_eur_mwh
            grid = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("UTC")
            if not test.index.equals(grid) or not np.isfinite(test).all():
                raise ValueError("common population incomplete")
            parent = test.resample("h").mean()
            raw, _ = calendar_cell_reference(train.price_eur_mwh.resample("h").mean(), parent.index)
            forecast = pd.Series(raw, index=parent.index)
            cal = enrich_15min_index(grid, country="DE")
            additive, _ = intraday_cell_reference(train.price_eur_mwh, grid, country="DE")
            frame = pd.DataFrame({"actual": test, "parent": parent.reindex(grid.floor("h")).to_numpy(),
                                  "forecast": forecast.reindex(grid.floor("h")).to_numpy(), "additive": additive})
            for lane, level in (("conditional", parent), ("forecast", forecast)):
                for candidate in CANDIDATES:
                    delta = (pd.Series(0., index=grid) if candidate == "flat" else additive if candidate == "signed-additive"
                             else models[candidate].price_conditioned_residual(level, grid, cal, reference_date=origin))
                    frame[f"{lane}/{candidate}"] = level.reindex(grid.floor("h")).to_numpy() + delta.to_numpy()
                    for segment, mask in segments(frame.parent.to_numpy(), frame.forecast.to_numpy(), grid).items():
                        error = (frame[f"{lane}/{candidate}"] - test).loc[mask]
                        metrics.append(dict(origin=month, delivery=delivery, lead=j-i, lane=lane,
                            candidate=candidate, segment=segment, rows=len(error),
                            mae=float(error.abs().mean()) if len(error) else None,
                            rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                            bias=float(error.mean()) if len(error) else None))
            folder = out / f"de-{month}-{delivery}"
            folder.mkdir()
            frame.to_parquet(folder / "predictions.parquet")
            if i == j:
                previous = pd.read_parquet(PREVIOUS / f"de-{month}/predictions.parquet")
                for candidate in CANDIDATES:
                    error = float((frame[f"conditional/{candidate}"] - frame.parent - previous[candidate]).abs().max())
                    if error > 1e-9:
                        raise ValueError("D305 same-population control changed")
                    controls.append(dict(origin=month, candidate=candidate, error=error))
        print(json.dumps(dict(stage="DE_origin_complete", origin=month)), flush=True)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)

    # Keep the signed-only hourly solution as the conditioning price authority.
    current_manifest = json.loads((SOURCE / "curve-final/manifest.json").read_text())
    reference = pd.Timestamp(current_manifest["valuation_at_utc"])
    hourly_model = HydroAlignedShapeHourlyMLP.load(SOURCE / "fitted-models/hourly.pkl")
    receipts, fit_records = [], []
    contexts = [(spec["id"], pd.Timestamp(spec["origin_utc"])) for spec in prior_plan["folds"]] + [("current", reference)]
    for label, origin in contexts:
        current = label == "current"
        baseline = pd.read_parquet(PREVIOUS / ("current/signed/curve.parquet" if current else f"{label}/signed/curve.parquet"))
        if current:
            solver = json.loads((SOURCE / "monthly-solver/result.json").read_text())
            levels = solver["assembler_base_prices"]
            surface = select_latest_quote_surface(pd.read_parquet(SOURCE / "eex-replay/eex-normalized-history.parquet"))
        else:
            solver = json.loads((OLD / f"{label}/solver.json").read_text())
            levels = solver["base_prices"]
            surface = pd.read_parquet(OLD / f"{label}/eex-surface.parquet")
        folder = out / label
        folder.mkdir()
        train = de.loc[de.index < origin]
        # Reject partial last hours at non-midnight historical origins.
        count = train.price_eur_mwh.groupby(train.index.floor("h")).transform("count")
        train = train.loc[count.eq(4)]
        fit_records.append(dict(origin=label, rows=len(train), cutoff=origin.isoformat(),
            last_training=train.index[-1].isoformat() if len(train) else None,
            status="FITTED" if len(train) else "UNSUPPORTED_FLAT_NO_DE_HISTORY"))
        for candidate, regularize in (("native", False), ("native-regularized", True)):
            model = ShapeIntraday(enable_sparse_support_regularization=regularize)
            if len(train):
                model.fit(train, None, enrich_15min_index(train.index, country="DE"))
                model.save(folder / f"{candidate}.parquet")
            grid = baseline.index
            delta = model.price_conditioned_residual(baseline.price_shape.resample("h").mean(), grid,
                enrich_15min_index(grid), reference_date=origin)
            assembly = assembler(hourly_model)
            frame = assembly.build(base_prices=levels, quoted_keys=set(solver["quoted_keys"]),
                delivery_index=grid, reference_date=origin, country="CH",
                signed_hourly_shape=baseline.signed_hourly_shape_eur_mwh.iloc[::4], signed_intraday_shape=delta)
            receipt = save_curve(folder / candidate, frame, levels, surface, assembly)
            drift = float((frame.price_shape.resample("h").mean() - baseline.price_shape.resample("h").mean()).abs().max())
            residual_error = float((frame.price_shape - baseline.price_shape - delta).abs().max())
            if max(drift, residual_error) > 1e-9:
                raise ValueError("composition changed signed hourly reference")
            receipts.append(dict(origin=label, candidate=candidate, hourly_drift=drift,
                                 residual_error=residual_error, **receipt))
            if current:
                main = frame.loc[grid < pd.Timestamp("2030-01-01", tz="Europe/Zurich")]
                pd.DataFrame({"timestamp_utc": main.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "timestamp_ch": main.index.tz_convert("Europe/Zurich").map(lambda t: t.isoformat()),
                    "price_eur_mwh": main.price_shape.to_numpy()}).to_csv(folder / candidate / "pfc-fmv-ch-15min.csv",
                        sep=";", index=False, float_format="%.10f")
        print(json.dumps(dict(stage="CH_composition_complete", origin=label, training_rows=len(train))), flush=True)
    write_json(out / "complete.json", dict(controls=controls, receipts=receipts, fits=fit_records,
        authority=dict(AUTHORITIES), status="COMPLETE_LOCAL_NO_ADOPTION", completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat()))
    for name, digest in {**pins, **code}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"bound bytes changed during execution: {name}")
    logging.shutdown()
    write_json(out / "manifest.json", {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()})


if __name__ == "__main__":
    main()
