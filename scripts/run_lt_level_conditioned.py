"""Freeze, then execute D318 in separate invocations; bounded local CPU only."""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.lt.benchmark_safeguards import ForbiddenHourlyModel, require_common_population, screen_segments
from pfc_shaping.lt.level_conditioned_shape import design, fit
from pfc_shaping.lt.local_benchmark import AUTHORITIES, score_curves
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from scripts.run_lt_hourly_stability import training_thresholds, stability_masks
from scripts.run_lt_hourly_recency import diagnostic_frame, stats
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, assembler, save_curve, sha, write_json

WORK = ROOT/'build/lt-level-conditioned-20260908'
PRIOR = ROOT/'build/lt-hourly-stability-20260908/run-v1'
CANDIDATES = {'signed-equal': None, 'level-ridge1': 1., 'level-ridge10': 10.}


def solver_for(label):
    path = SOURCE/'monthly-solver/result.json' if label == 'current' else OLD/label/'solver.json'
    solver = json.loads(path.read_text())
    return solver, solver['assembler_base_prices'] if label == 'current' else solver['base_prices']


def prepare_pairs(spec, specs):
    origin = pd.Timestamp(spec['origin_utc'])
    target = pd.read_parquet(PRIOR/spec['id']/'targets.parquet')
    parts = []
    for inner in specs:
        at = pd.Timestamp(inner['origin_utc'])
        if at >= origin:
            continue
        raw = pd.read_parquet(PRIOR/inner['id']/'signed-equal/raw.parquet').raw
        index = raw.index.intersection(target.index)
        if index.empty:
            continue
        _, levels = solver_for(inner['id'])
        # Select whole closed target months; never substitute realized means for B.
        month = index.tz_convert('Europe/Zurich').strftime('%Y-%m')
        base = center_signed_hourly_shape(raw.reindex(index))
        parts.append(pd.DataFrame(dict(origin=at, delivery=index,
            level=[levels[m] for m in month], baseline=base.to_numpy(),
            residual=target.signed_target.reindex(index).to_numpy()-base.to_numpy())))
    if not parts:
        return pd.DataFrame(columns=['origin', 'delivery', 'level', 'baseline', 'residual', 'weight'])
    pairs = pd.concat(parts, ignore_index=True)
    pairs['weight'] = 1/pairs.groupby('delivery').delivery.transform('size')
    return pairs


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((PRIOR/'plan.json').read_text())
    pins = dict(old['inputs_sha256'])
    for name, digest in json.loads((PRIOR/'manifest.json').read_text()).items():
        pins[(PRIOR/name).relative_to(ROOT).as_posix()] = digest
    preservation = json.loads((ROOT/'build/lt-audit-response-20260908/prior-artifacts-verification.json').read_text())
    pins.update(preservation['files_sha256'])
    # Bind all pilot files as well, without executing the daily collector.
    for path in (ROOT/'build/lt-matched-vintages-20260908/registry').rglob('*'):
        if path.is_file():
            pins[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, digest in pins.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'prior input changed: {name}')
    code_names = set(old['code_sha256']) | {
        'scripts/run_lt_level_conditioned.py', 'scripts/verify_lt_level_conditioned.py',
        'pfc_shaping/lt/level_conditioned_shape.py', 'pfc_shaping/lt/benchmark_safeguards.py',
        'scripts/verify_lt_maturity.py', 'scripts/verify_lt_hourly_recency.py',
        'scripts/verify_lt_hourly_stability.py', 'scripts/lt_maturity_experiment.py',
        'docs/model/LT-LEVEL-CONDITIONED-EXPERIMENT-20260908.md',
        'tests/test_benchmark_safeguards.py', 'tests/test_level_conditioned_shape.py'}
    code = {name: sha(ROOT/name) for name in sorted(code_names)}
    for name in code:
        dest = out/'source-snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    thresholds, pair_pins, support = {}, {}, []
    for spec in old['origins']:
        label = spec['id']
        target = pd.read_parquet(PRIOR/label/'targets.parquet')
        thresholds[label] = training_thresholds(target, pd.Timestamp(spec['origin_utc']))
        assert thresholds[label] == old['thresholds'][label]
        pairs = prepare_pairs(spec, old['origins'])
        path = out/f'pairs-{label}.parquet'
        pairs.to_parquet(path, index=False)
        pair_pins[path.name] = sha(path)
        support.append(dict(origin=label, pairs=len(pairs),
            inner_origins=int(pairs.origin.nunique()), distinct_hours=int(pairs.delivery.nunique()),
            status='SUPPORTED_RETROSPECTIVE' if len(pairs) else 'UNSUPPORTED_NO_PAIRS'))
    write_json(out/'plan.json', dict(schema='fmv-level-conditioned-shape.v1',
        frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(), candidates=CANDIDATES,
        origins=old['origins'], thresholds=thresholds, support=support,
        evidence_class='EXPOSED_DEVELOPMENT_NO_HOLDOUT',
        exposed_origins=[s['id'] for s in old['origins']],
        source_policy='ARCHIVED_SOLVER_RECONSTRUCTIONS_LATEST_OBSERVED_NOT_PIT',
        max_fits=14, max_assemblies=21, threads=4, device='CPU',
        inputs_sha256=pins, code_sha256=code, pairs_sha256=pair_pins,
        authority=dict(AUTHORITIES), retained_reference='D304',
        comparison_policy='GLOBAL_RECIPES_NO_MONTH_SELECTION',
        future_holdout='2026-10/2027-09_DRAFT_NOT_INDEPENDENTLY_REGISTERED'))
    write_json(out/'freeze-receipt.json', dict(plan_sha256=sha(out/'plan.json'), results_exist=False))
    print(json.dumps(dict(stage='FROZEN_BEFORE_FIT', plan_sha256=sha(out/'plan.json'), support=support)), flush=True)


def validate(out):
    plan = json.loads((out/'plan.json').read_text())
    assert sha(out/'plan.json') == json.loads((out/'freeze-receipt.json').read_text())['plan_sha256']
    for name, digest in {**plan['inputs_sha256'], **plan['code_sha256']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError(f'frozen bytes changed: {name}')
    for name, digest in plan['pairs_sha256'].items():
        assert sha(out/name) == digest
    return plan


def run(out):
    plan = validate(out)
    result = out/'results'
    result.mkdir(exist_ok=False)
    logging.basicConfig(filename=result/'runtime.log', level=logging.INFO)
    truth = pd.read_parquet(SOURCE/'prepared-inputs/epex-ch.parquet').price_eur_mwh
    grouped = truth.resample('h')
    assert grouped.count().eq(4).all() and (grouped.max()-grouped.min()).eq(0).all()
    truth = truth.loc[truth.index < pd.Timestamp('2026-09-01', tz='Europe/Zurich')]
    records, receipts, coverage_rows, support = [], [], [], []
    fits = assemblies = 0
    for spec in plan['origins']:
        label, origin = spec['id'], pd.Timestamp(spec['origin_utc'])
        solver, levels = solver_for(label)
        surface = (select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet'))
                   if label == 'current' else pd.read_parquet(OLD/label/'eex-surface.parquet'))
        baseline = pd.read_parquet(PRIOR/label/'signed-equal/curve.parquet')
        reference = pd.read_parquet(PRIOR/label/'signed-equal/raw.parquet').raw
        month = reference.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
        level = pd.Series([levels[m] for m in month], index=reference.index)
        x = design(reference.index, level)
        pairs = pd.read_parquet(out/f'pairs-{label}.parquet')
        folder = result/label
        folder.mkdir()
        if len(pairs):
            outside = (level < pairs.level.min()) | (level > pairs.level.max())
            support.append(dict(origin=label, extrapolated_level_hours=int(outside.sum()), total_hours=len(level)))
        else:
            support.append(dict(origin=label, extrapolated_level_hours=len(level), total_hours=len(level)))
        populations = {}
        for candidate, ridge in plan['candidates'].items():
            beta = np.zeros(54) if ridge is None else fit(pairs, origin, ridge)
            fits += int(ridge is not None and not pairs.empty)
            raw = reference.copy() if ridge is None else reference+pd.Series(x@beta, index=reference.index)
            assembly = assembler(ForbiddenHourlyModel())
            frame = assembly.build(base_prices=levels, quoted_keys=set(solver['quoted_keys']),
                delivery_index=baseline.index, reference_date=origin, country='CH',
                signed_hourly_shape=center_signed_hourly_shape(raw))
            assemblies += 1
            assert frame.f_H.eq(1).all()
            if ridge is None:
                np.testing.assert_allclose(frame.price_shape, baseline.price_shape, atol=1e-9, rtol=0)
                np.testing.assert_array_equal(raw, reference)
            dest = folder/candidate
            receipts.append(dict(origin=label, candidate=candidate,
                **save_curve(dest, frame, levels, surface, assembly)))
            raw.to_frame('raw').to_parquet(dest/'raw.parquet')
            write_json(dest/'fit.json', dict(beta=beta.tolist(), ridge=ridge,
                training_pairs=len(pairs), fitted=bool(ridge is not None and len(pairs))))
            if label != 'current':
                for stage, column in [('final', 'price_shape'), ('pre_projection', 'price_pre_final_projection')]:
                    _, errors, coverage = score_curves(frame[[column]], truth, origin)
                    require_common_population(populations.setdefault('all', errors.index), errors.index, origin)
                    coverage_rows.extend(dict(origin=label, candidate=candidate, stage=stage, **c) for c in coverage)
                    actual = truth.resample('h').mean().reindex(errors.index)
                    diagnostic = diagnostic_frame(frame[column].resample('h').mean().reindex(errors.index), actual)
                    diagnostic.to_parquet(dest/f'diagnostics-{stage}.parquet')
                    for segment, mask in stability_masks(diagnostic, origin, plan['thresholds'][label]).items():
                        for field in ['shape_error', 'full_error', 'level_error', 'ramp_error']:
                            records.append(dict(origin=label, role=spec['role'], candidate=candidate,
                                stage=stage, segment=segment, error_type=field, **stats(diagnostic.loc[mask, field])))
            else:
                for grain, series in [('1h', frame.price_shape.resample('h').mean()), ('15min', frame.price_shape)]:
                    pd.DataFrame(dict(timestamp_utc=series.index.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        timestamp_ch=series.index.tz_convert('Europe/Zurich').map(lambda t: t.isoformat()),
                        price_eur_mwh=series.to_numpy())).to_csv(dest/f'pfc-experimental-{grain}.csv',
                        sep=';', index=False, float_format='%.10f')
        print(json.dumps(dict(stage='origin_complete', origin=label, fits=fits, assemblies=assemblies)), flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(result/'metrics.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(result/'coverage.csv', index=False)
    pd.DataFrame(support).to_csv(result/'level-extrapolation.csv', index=False)
    assessment = metrics.loc[metrics.role.eq('assessment')]
    gates, decision = screen_segments(assessment, list(CANDIDATES)[1:])
    gates.to_csv(result/'origin-segment-gates.csv', index=False)
    write_json(result/'decision.json', dict(selections=decision, retained_reference='D304',
        evidence_class=plan['evidence_class'], authority=dict(AUTHORITIES)))
    assert fits <= plan['max_fits'] and assemblies == 21
    validate(out)
    write_json(result/'complete.json', dict(fits=fits, assemblies=assemblies, receipts=receipts,
        completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat(), authority=dict(AUTHORITIES)))
    logging.shutdown()
    write_json(result/'manifest.json', {p.relative_to(result).as_posix(): sha(p) for p in result.rglob('*') if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if Path.cwd() != ROOT or not out.is_relative_to(WORK) or out == WORK:
        raise ValueError('canonical fresh task paths required')
    for name in ('TEMP', 'TMP', 'APPDATA', 'LOCALAPPDATA', 'MPLCONFIGDIR', 'XDG_CACHE_HOME',
                 'NUMBA_CACHE_DIR', 'JOBLIB_TEMP_FOLDER', 'PYTHONUSERBASE', 'PIP_CACHE_DIR'):
        if not Path(os.environ.get(name, '')).resolve().is_relative_to(WORK):
            raise ValueError('task-local runtime required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise ValueError('CPU required')
    {'freeze': freeze, 'run': run}[args.action](out)


if __name__ == '__main__':
    main()
