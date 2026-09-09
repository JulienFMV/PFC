"""D308 fixed global signed-shape blends; local CPU, no operational authority."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from pfc_shaping.lt.local_benchmark import AUTHORITIES, score_curves
from pfc_shaping.lt.benchmark_safeguards import ForbiddenHourlyModel
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from scripts.run_lt_hourly_recency import diagnostic_frame, masks, stats
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, assembler, save_curve, sha, write_json

WORK = ROOT / 'build/lt-hourly-stability-20260908'
PRIOR = ROOT / 'build/lt-hourly-recency-20260908/run-v2'
CONTROLS = ('mlp-control', 'signed-equal', 'signed-hl365', 'signed-hl730')
BLENDS = {f'blend-hl{half}-a{pct}': dict(component=f'signed-hl{half}', alpha=pct/100)
          for half in (365, 730) for pct in (25, 50)}


def blend_raw(reference, recent, alpha):
    if (not reference.index.equals(recent.index) or not reference.index.is_unique
            or not np.isfinite(reference).all() or not np.isfinite(recent).all()
            or isinstance(alpha, bool) or not np.isscalar(alpha)
            or not np.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError('aligned finite signed shapes and scalar convex weight required')
    return (1-alpha)*reference + alpha*recent


def training_thresholds(target, origin):
    if target.empty or target.index.max() >= origin:
        raise ValueError('nonempty pre-origin training required')
    months = target.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    for month in set(months):
        start = pd.Timestamp(month+'-01', tz='Europe/Zurich')
        end = (pd.Period(month, freq='M')+1).start_time.tz_localize('Europe/Zurich')
        if end > origin or not target.loc[months == month].index.equals(
                pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC')):
            raise ValueError('complete closed training months required')
    price = target.price_eur_mwh
    signed = price-price.groupby(months).transform('mean')
    if not np.isfinite(price).all() or not np.allclose(signed, target.signed_target, atol=1e-12, rtol=0):
        raise ValueError('finite monthly signed targets required')
    ramp = diagnostic_frame(price, price).truth_ramp
    return dict(price_p95=float(price.quantile(.95)), ramp_p95=float(ramp.abs().quantile(.95)),
                shape_p05=float(signed.quantile(.05)), shape_p95=float(signed.quantile(.95)),
                shape_abs_p95=float(signed.abs().quantile(.95)))


def stability_masks(frame, origin, thresholds):
    result = masks(frame, origin, thresholds)
    months = frame.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    shape = frame.actual-frame.actual.groupby(months).transform('mean')
    result.update(SHAPE_LOW=shape.to_numpy()<thresholds['shape_p05'],
                  SHAPE_HIGH=shape.to_numpy()>thresholds['shape_p95'],
                  SHAPE_ABS_TAIL=shape.abs().to_numpy()>thresholds['shape_abs_p95'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if Path.cwd() != ROOT or not out.is_relative_to(WORK) or out == WORK:
        raise ValueError('fresh canonical task output required')
    for name in ('TEMP', 'TMP', 'APPDATA', 'LOCALAPPDATA', 'MPLCONFIGDIR', 'XDG_CACHE_HOME',
                 'NUMBA_CACHE_DIR', 'JOBLIB_TEMP_FOLDER', 'PYTHONUSERBASE', 'PIP_CACHE_DIR'):
        if not Path(os.environ.get(name, '')).resolve().is_relative_to(WORK):
            raise ValueError(f'task-local runtime required: {name}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise ValueError('CPU required')
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=out/'runtime.log', level=logging.INFO)
    if sha(PRIOR/'manifest.json') != '05a44fbda7f3619d129a88943e68cf76950ac3564ad25c0fc1b286023152d8ab':
        raise ValueError('D307 manifest mismatch')
    previous = json.loads((PRIOR/'plan.json').read_text())
    pins = dict(previous['inputs_sha256'])
    pins[(PRIOR/'manifest.json').relative_to(ROOT).as_posix()] = sha(PRIOR/'manifest.json')
    for name, digest in json.loads((PRIOR/'manifest.json').read_text()).items():
        pins[(PRIOR/name).relative_to(ROOT).as_posix()] = digest
    code = dict(previous['code_sha256'])
    for name, digest in {**pins, **code}.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'prior bytes changed: {name}')
    for name in ('scripts/run_lt_hourly_stability.py', 'scripts/verify_lt_hourly_stability.py',
                 'scripts/verify_lt_hourly_recency.py', 'tests/test_hourly_stability.py',
                 'docs/model/LT-HOURLY-STABILITY-LOCAL-EXPERIMENT.md'):
        code[name] = sha(ROOT/name)
    for name in code:
        dest = out/'source-snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    reference = pd.Timestamp(json.loads((SOURCE/'curve-final/manifest.json').read_text())['valuation_at_utc'])
    specs = previous['folds']+[dict(id='current', role='descriptive', origin_utc=reference.isoformat())]
    thresholds = {s['id']: training_thresholds(pd.read_parquet(PRIOR/s['id']/'targets.parquet'),
                                               pd.Timestamp(s['origin_utc'])) for s in specs}
    plan = dict(schema='fmv-hourly-stability.v1', frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                candidates=BLENDS, controls=CONTROLS, folds=previous['folds'], origins=specs,
                thresholds=thresholds, inputs_sha256=pins, code_sha256=code, authority=dict(AUTHORITIES),
                max_new_assemblies=28, statistical_estimator_fits=0,
                valuation='RETAINED_20260907_NOT_REPRICED')
    write_json(out/'plan.json', plan)
    print(json.dumps(dict(stage='plan_frozen', sha256=sha(out/'plan.json'))), flush=True)
    truth = pd.read_parquet(SOURCE/'prepared-inputs/epex-ch.parquet').price_eur_mwh
    truth = truth.loc[truth.index<pd.Timestamp('2026-09-01', tz='Europe/Zurich')]
    model = ForbiddenHourlyModel()
    scores, diagnostics, monthly, receipts = [], [], [], []
    for spec in specs:
        label, origin = spec['id'], pd.Timestamp(spec['origin_utc'])
        folder = out/label
        folder.mkdir()
        shutil.copyfile(PRIOR/label/'targets.parquet', folder/'targets.parquet')
        write_json(folder/'thresholds.json', thresholds[label])
        solver = json.loads((SOURCE/'monthly-solver/result.json' if label=='current' else OLD/f'{label}/solver.json').read_text())
        levels = solver['assembler_base_prices'] if label=='current' else solver['base_prices']
        surface = (select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet'))
                   if label=='current' else pd.read_parquet(OLD/f'{label}/eex-surface.parquet'))
        baseline = pd.read_parquet(PRIOR/label/'signed-equal/curve.parquet')
        grid, population = baseline.index, None
        for candidate in list(CONTROLS)+list(BLENDS):
            dest = folder/candidate
            if candidate in CONTROLS:
                shutil.copytree(PRIOR/label/candidate, dest)
                frame = pd.read_parquet(dest/'curve.parquet')
            else:
                config = BLENDS[candidate]
                raw = blend_raw(pd.read_parquet(PRIOR/label/'signed-equal/raw.parquet').raw,
                                pd.read_parquet(PRIOR/label/config['component']/'raw.parquet').raw, config['alpha'])
                assembly = assembler(model)
                frame = assembly.build(base_prices=levels, quoted_keys=set(solver['quoted_keys']), delivery_index=grid,
                                       reference_date=origin, country='CH', signed_hourly_shape=center_signed_hourly_shape(raw))
                receipts.append(dict(origin=label, candidate=candidate, **save_curve(dest, frame, levels, surface, assembly)))
                raw.to_frame('raw').to_parquet(dest/'raw.parquet')
                write_json(dest/'blend.json', config)
            if label=='current':
                if candidate in BLENDS:
                    main = frame.loc[grid<pd.Timestamp('2030-01-01', tz='Europe/Zurich')]
                    for resolution, series in [('1h', main.price_shape.resample('h').mean()), ('15min', main.price_shape)]:
                        pd.DataFrame(dict(timestamp_utc=series.index.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                          timestamp_ch=series.index.tz_convert('Europe/Zurich').map(lambda t:t.isoformat()),
                                          price_eur_mwh=series.to_numpy())).to_csv(dest/f'pfc-fmv-ch-{resolution}.csv', sep=';', index=False, float_format='%.10f')
                continue
            for stage, column in [('final', 'price_shape'), ('pre_projection', 'price_pre_final_projection')]:
                metrics, errors, _ = score_curves(frame[[column]].rename(columns={column:candidate}), truth, origin)
                if population is None:
                    population = errors.index
                if not population.equals(errors.index):
                    raise ValueError('unpaired populations')
                errors.to_parquet(dest/f'errors-{stage}.parquet')
                metrics['origin'], metrics['role'], metrics['stage'] = label, spec['role'], stage
                scores.append(metrics)
                diagnostic = diagnostic_frame(frame[column].resample('h').mean().reindex(population), truth.resample('h').mean().reindex(population))
                np.testing.assert_allclose(diagnostic.shape_error, errors[candidate], atol=1e-10, rtol=0)
                diagnostic.to_parquet(dest/f'diagnostics-{stage}.parquet')
                for segment, mask in stability_masks(diagnostic, origin, thresholds[label]).items():
                    for field in ['shape_error', 'full_error', 'level_error', 'ramp_error']:
                        diagnostics.append(dict(origin=label, role=spec['role'], candidate=candidate, stage=stage,
                                                segment=segment, error_type=field, **stats(diagnostic.loc[mask, field])))
                for month, part in diagnostic.groupby(population.tz_convert('Europe/Zurich').strftime('%Y-%m')):
                    monthly.append(dict(origin=label, candidate=candidate, stage=stage, month=month, hours=len(part),
                                        predicted_mean=float(part.prediction.mean()), actual_mean=float(part.actual.mean()),
                                        shape_mse=float(np.mean(part.shape_error**2)), level_mse=float(np.mean(part.level_error**2)),
                                        full_mse=float(np.mean(part.full_error**2))))
        print(json.dumps(dict(stage='origin_complete', origin=label)), flush=True)
    pd.concat(scores).to_csv(out/'native-metrics.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostic-metrics.csv', index=False)
    pd.DataFrame(monthly).to_csv(out/'monthly-decomposition.csv', index=False)
    for name, digest in {**pins, **code}.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'bound bytes changed: {name}')
    assert len(receipts)==28
    write_json(out/'complete.json', dict(status='COMPLETE_LOCAL_NO_ADOPTION', receipts=receipts,
               new_assemblies=len(receipts), cached_control_curves=28, statistical_estimator_fits=0,
               authority=dict(AUTHORITIES), completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    logging.shutdown()
    write_json(out/'manifest.json', {p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})


if __name__=='__main__':
    main()
