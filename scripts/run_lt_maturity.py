"""Run the pre-specified D310 residual maturity experiment."""
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
from pfc_shaping.lt.benchmark_safeguards import ForbiddenHourlyModel, require_common_population
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from scripts.lt_maturity_experiment import basis, pairs, fit, leads
from scripts.run_lt_hourly_stability import stability_masks, training_thresholds
from scripts.run_lt_hourly_recency import diagnostic_frame, stats
from scripts.audit_lt_hourly_revisions import preorigin_scales, components, event_frame, event_summary, revision_frame
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, assembler, save_curve, sha, write_json

WORK = ROOT/'build/lt-maturity-20260908'
PRIOR = ROOT/'build/lt-hourly-stability-20260908/run-v1'
CANDIDATES = ('signed-equal', 'residual-calendar', 'residual-maturity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    if Path.cwd() != ROOT or not out.is_relative_to(WORK) or out == WORK:
        raise ValueError('fresh canonical task output required')
    for name in ('TEMP', 'TMP', 'APPDATA', 'LOCALAPPDATA', 'MPLCONFIGDIR', 'XDG_CACHE_HOME',
                 'NUMBA_CACHE_DIR', 'JOBLIB_TEMP_FOLDER', 'PYTHONUSERBASE', 'PIP_CACHE_DIR'):
        if not Path(os.environ.get(name, '')).resolve().is_relative_to(WORK):
            raise ValueError('task-local runtime required')
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=out/'runtime.log', level=logging.INFO)
    if sha(PRIOR/'manifest.json') != '1f131effbce0af30e418aa9f47d72bb097c04746e23c7cd68feec9ffeecb0628':
        raise ValueError('D308 manifest mismatch')
    previous = json.loads((PRIOR/'plan.json').read_text())
    pins = dict(previous['inputs_sha256'])
    pins[(PRIOR/'manifest.json').relative_to(ROOT).as_posix()] = sha(PRIOR/'manifest.json')
    for name, digest in json.loads((PRIOR/'manifest.json').read_text()).items():
        pins[(PRIOR/name).relative_to(ROOT).as_posix()] = digest
    code = dict(previous['code_sha256'])
    for name in ('scripts/lt_maturity_experiment.py', 'scripts/run_lt_maturity.py',
                 'scripts/verify_lt_maturity.py', 'tests/test_lt_maturity.py',
                 'docs/model/LT-MATURITY-LOCAL-EXPERIMENT.md', 'scripts/audit_lt_hourly_revisions.py'):
        code[name] = sha(ROOT/name)
    for name, digest in {**pins, **code}.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'bound bytes changed: {name}')
    for name in code:
        dest = out/'source-snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    specs = previous['origins']
    scales = {s['id']: preorigin_scales(pd.read_parquet(PRIOR/s['id']/'targets.parquet'),
                                       pd.Timestamp(s['origin_utc'])) for s in specs}
    plan = dict(schema='fmv-origin-delivery-maturity.v1', frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        candidates=CANDIDATES, origins=specs, thresholds=previous['thresholds'], scales=scales,
        inputs_sha256=pins, code_sha256=code, authority=dict(AUTHORITIES), max_fits=14,
        max_new_assemblies=14, gpu_authorized=True, device='CPU', threads=4,
        ridge=.1, pair_cadence='QUARTERLY_MINUS_12H', min_history_months=12, max_lead=36,
        weight='INVERSE_PAIRS_PER_DELIVERY_HOUR', feature_dimensions=[54, 108],
        source_policy='LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT', valuation='RETAINED_20260907_NOT_REPRICED')
    write_json(out/'plan.json', plan)
    print(json.dumps(dict(stage='plan_frozen', sha256=sha(out/'plan.json'))), flush=True)
    truth = pd.read_parquet(SOURCE/'prepared-inputs/epex-ch.parquet').price_eur_mwh
    truth = truth.loc[truth.index < pd.Timestamp('2026-09-01', tz='Europe/Zurich')]
    model = ForbiddenHourlyModel()
    metrics_rows, events, revisions, supports, receipts = [], [], [], [], []
    previous_parts = {}
    for spec in specs:
        label, origin = spec['id'], pd.Timestamp(spec['origin_utc'])
        folder = out/label
        folder.mkdir()
        target = pd.read_parquet(PRIOR/label/'targets.parquet')
        training_thresholds(target, origin)
        pair_frame = pairs(target, origin)
        pair_frame.to_parquet(folder/'pairs.parquet', index=False)
        support = pair_frame.groupby('lead').agg(pairs=('delivery', 'size'), distinct_hours=('delivery', 'nunique'))
        support.to_csv(folder/'training-support.csv')
        write_json(folder/'training.json', dict(pairs=len(pair_frame), distinct_hours=int(pair_frame.delivery.nunique()),
            information_origins=int(pair_frame.origin.nunique()), min_lead=int(pair_frame.lead.min()),
            max_lead=int(pair_frame.lead.max()), weight_sum=float(pair_frame.weight.sum())))
        solver = json.loads((SOURCE/'monthly-solver/result.json' if label == 'current' else OLD/f'{label}/solver.json').read_text())
        levels = solver['assembler_base_prices'] if label == 'current' else solver['base_prices']
        surface = (select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet'))
                   if label == 'current' else pd.read_parquet(OLD/f'{label}/eex-surface.parquet'))
        baseline = pd.read_parquet(PRIOR/label/'signed-equal/curve.parquet')
        reference = pd.read_parquet(PRIOR/label/'signed-equal/raw.parquet').raw
        delivery_leads = leads(reference.index, origin)
        supports.append(dict(origin=label, hours=len(reference), training_max_lead=int(pair_frame.lead.max()),
            unsupported_hours=int((delivery_leads > pair_frame.lead.max()).sum()),
            beyond_36_hours=int((delivery_leads > 36).sum())))
        populations = {}
        for candidate in CANDIDATES:
            dest = folder/candidate
            raw = reference.copy()
            if candidate == 'signed-equal':
                shutil.copytree(PRIOR/label/candidate, dest)
                frame = baseline
            else:
                maturity = candidate == 'residual-maturity'
                beta = fit(pair_frame, maturity)
                correction = basis(reference.index, origin, maturity)@beta
                raw += correction
                assembly = assembler(model)
                frame = assembly.build(base_prices=levels, quoted_keys=set(solver['quoted_keys']),
                    delivery_index=baseline.index, reference_date=origin, country='CH',
                    signed_hourly_shape=center_signed_hourly_shape(raw))
                receipts.append(dict(origin=label, candidate=candidate, **save_curve(dest, frame, levels, surface, assembly)))
                raw.to_frame('raw').to_parquet(dest/'raw.parquet')
                write_json(dest/'fit.json', dict(coefficients=beta.tolist(), maturity=maturity, ridge=.1))
            actual = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz='UTC'))
            if label != 'current':
                for stage, column in [('final', 'price_shape'), ('pre_projection', 'price_pre_final_projection')]:
                    _, errors, _ = score_curves(frame[[column]].rename(columns={column: candidate}), truth, origin)
                    require_common_population(populations.setdefault(stage, errors.index), errors.index, origin)
                    actual = truth.resample('h').mean().reindex(errors.index)
                    diagnostic = diagnostic_frame(frame[column].resample('h').mean().reindex(errors.index), actual)
                    diagnostic.to_parquet(dest/f'diagnostics-{stage}.parquet')
                    for segment, mask in stability_masks(diagnostic, origin, plan['thresholds'][label]).items():
                        for field in ('shape_error', 'full_error', 'level_error', 'ramp_error'):
                            metrics_rows.append(dict(origin=label, role=spec['role'], candidate=candidate,
                                stage=stage, segment=segment, error_type=field, **stats(diagnostic.loc[mask, field])))
            else:
                main = frame.loc[frame.index < pd.Timestamp('2030-01-01', tz='Europe/Zurich')]
                for resolution, series in [('1h', main.price_shape.resample('h').mean()), ('15min', main.price_shape)]:
                    pd.DataFrame(dict(timestamp_utc=series.index.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        timestamp_ch=series.index.tz_convert('Europe/Zurich').map(lambda t: t.isoformat()),
                        price_eur_mwh=series.to_numpy())).to_csv(dest/f'pfc-fmv-ch-{resolution}.csv', sep=';', index=False, float_format='%.10f')
            part = components(frame, levels)
            event = event_frame(part, raw, actual, origin, scales[label], plan['thresholds'][label])
            event['origin'], event['role'], event['candidate'] = label, spec['role'], candidate
            events.append(event)
            if candidate in previous_parts:
                old_label, older = previous_parts[candidate]
                delta = revision_frame(older, part)
                delta.to_parquet(dest/'revision-from-previous.parquet')
                revisions.append(dict(earlier=old_label, later=label, candidate=candidate, hours=len(delta),
                    status='RETROSPECTIVE_COMMON_DELIVERY' if len(delta) else 'UNSUPPORTED',
                    **{c+'_mae': float(delta[c].abs().mean()) if len(delta) else None for c in delta}))
            previous_parts[candidate] = (label, part)
        print(json.dumps(dict(stage='origin_complete', origin=label, pairs=len(pair_frame))), flush=True)
    pd.DataFrame(metrics_rows).to_csv(out/'diagnostic-metrics.csv', index=False)
    event_data = pd.concat(events)
    event_data.to_parquet(out/'events.parquet')
    event_summary(event_data).to_csv(out/'seam-metrics.csv', index=False)
    pd.DataFrame(revisions).to_csv(out/'revisions.csv', index=False)
    pd.DataFrame(supports).to_csv(out/'delivery-support.csv', index=False)
    for name, digest in {**pins, **code}.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'bound bytes changed: {name}')
    assert len(receipts) == 14
    write_json(out/'complete.json', dict(status='COMPLETE_LOCAL_NO_ADOPTION', fits=14, receipts=receipts,
        authority=dict(AUTHORITIES), completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    logging.shutdown()
    write_json(out/'manifest.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()})


if __name__ == '__main__':
    main()
