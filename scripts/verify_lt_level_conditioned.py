"""Independent saved-artifact arithmetic replay for D318; never refits selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, sha, write_json
from scripts.verify_lt_maturity import independent_basis
from scripts.verify_lt_hourly_recency import independent_masks
from pfc_shaping.validation.product_normalization import build_product_normalization_gates
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface

PRIOR = ROOT/'build/lt-hourly-stability-20260908/run-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    task = ROOT/'build/lt-level-conditioned-20260908'
    if Path.cwd() != ROOT or not run.is_relative_to(task) or not out.is_relative_to(task) or out == task:
        raise ValueError('canonical task paths required')
    out.mkdir(exist_ok=False)
    plan = json.loads((run/'plan.json').read_text())
    assert sha(run/'plan.json') == json.loads((run/'freeze-receipt.json').read_text())['plan_sha256']
    for name, digest in {**plan['inputs_sha256'], **plan['code_sha256']}.items():
        assert sha(ROOT/name) == digest, name
    for name, digest in plan['pairs_sha256'].items():
        assert sha(run/name) == digest
    result = run/'results'
    for name, digest in json.loads((result/'manifest.json').read_text()).items():
        assert sha(result/name) == digest, name
    truth_qh = pd.read_parquet(SOURCE/'prepared-inputs/epex-ch.parquet').price_eur_mwh
    truth = truth_qh.resample('h').mean()
    saved = pd.read_csv(result/'metrics.csv', dtype={'origin': str}).set_index(
        ['origin', 'candidate', 'stage', 'segment', 'error_type'])
    comparisons, replays = [], []
    max_level = max_beta = max_raw = 0.
    metrics_count = 0
    for spec in plan['origins']:
        label, origin = spec['id'], pd.Timestamp(spec['origin_utc'])
        target = pd.read_parquet(PRIOR/label/'targets.parquet')
        assert target.index.max() < origin
        target_months = target.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
        train_ramp = target.price_eur_mwh.diff()
        train_ramp.iloc[np.r_[True, target_months[1:] != target_months[:-1]]] = np.nan
        t = plan['thresholds'][label]
        np.testing.assert_allclose([t['price_p95'], t['ramp_p95'], t['shape_p05'], t['shape_p95'], t['shape_abs_p95']],
            [target.price_eur_mwh.quantile(.95), train_ramp.abs().quantile(.95),
             target.signed_target.quantile(.05), target.signed_target.quantile(.95), target.signed_target.abs().quantile(.95)], atol=1e-10, rtol=0)
        pair = pd.read_parquet(run/f'pairs-{label}.parquet')
        gram, rhs = np.zeros((54, 54)), np.zeros(54)
        if len(pair):
            assert (pair.delivery < origin).all() and (pair.origin < pair.delivery).all()
            assert not pair.duplicated(['origin', 'delivery']).any()
            np.testing.assert_allclose(pair.groupby('delivery').weight.sum(), 1, atol=1e-14)
            for inner, part in pair.groupby('origin'):
                inner_spec = next(s for s in plan['origins'] if pd.Timestamp(s['origin_utc']) == inner)
                inner_solver = json.loads((OLD/inner_spec['id']/'solver.json').read_text())['base_prices']
                index = pd.DatetimeIndex(part.delivery)
                month = index.tz_convert('Europe/Zurich').strftime('%Y-%m')
                level = np.array([inner_solver[m] for m in month])
                np.testing.assert_array_equal(part.level, level)
                raw = pd.read_parquet(PRIOR/inner_spec['id']/'signed-equal/raw.parquet').raw.reindex(index)
                baseline = raw-raw.groupby(month).transform('mean')
                np.testing.assert_allclose(part.baseline, baseline, rtol=0, atol=1e-10)
                np.testing.assert_allclose(part.residual, target.signed_target.reindex(index)-baseline, rtol=0, atol=1e-10)
                for m in set(month):
                    start = pd.Timestamp(m+'-01', tz='Europe/Zurich')
                    end = (pd.Period(m, freq='M')+1).start_time.tz_localize('Europe/Zurich')
                    assert end <= origin
                    assert index[month == m].equals(pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC'))
                x = independent_basis(index, inner, False)*level[:, None]/100
                gram += x.T@(part.weight.to_numpy()[:, None]*x)
                rhs += x.T@(part.weight.to_numpy()*part.residual.to_numpy())
        solver = json.loads((SOURCE/'monthly-solver/result.json' if label == 'current' else OLD/label/'solver.json').read_text())
        levels = solver['assembler_base_prices'] if label == 'current' else solver['base_prices']
        surface = (select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet'))
                   if label == 'current' else pd.read_parquet(OLD/label/'eex-surface.parquet'))
        population = None
        for candidate, ridge in plan['candidates'].items():
            dest = result/label/candidate
            fit = json.loads((dest/'fit.json').read_text())
            beta = np.zeros(54)
            if ridge is not None and len(pair):
                eigenvalues, eigenvectors = np.linalg.eigh(gram/pair.weight.sum())
                beta = eigenvectors@((eigenvectors.T@(rhs/pair.weight.sum()))/(eigenvalues+ridge))
            max_beta = max(max_beta, float(np.max(np.abs(beta-fit['beta']))))
            np.testing.assert_allclose(beta, fit['beta'], atol=1e-9, rtol=0)
            raw = pd.read_parquet(dest/'raw.parquet').raw
            ref = pd.read_parquet(PRIOR/label/'signed-equal/raw.parquet').raw
            assert raw.index.equals(ref.index)
            level = np.array([levels[m] for m in raw.index.tz_convert('Europe/Zurich').strftime('%Y-%m')])
            expected = ref+independent_basis(raw.index, origin, False)@beta*level/100
            max_raw = max(max_raw, float(np.max(np.abs(raw-expected))))
            np.testing.assert_allclose(raw, expected, atol=1e-9, rtol=0)
            frame = pd.read_parquet(dest/'curve.parquet')
            hourly = frame[['price_shape', 'price_pre_final_projection']].resample('h').mean()
            months = hourly.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
            for column in hourly:
                means = hourly[column].groupby(months).mean()
                err = max(abs(means[m]-levels[m]) for m in means.index)
                max_level = max(max_level, err)
                assert err < 1e-9
            expected_pre = raw-raw.groupby(months).transform('mean')+level
            np.testing.assert_allclose(hourly.price_pre_final_projection, expected_pre, atol=1e-9, rtol=0)
            local = hourly.index.tz_convert('Europe/Zurich')
            peak = (local.dayofweek < 5) & (local.hour >= 8) & (local.hour < 20)
            delta = hourly.price_shape-hourly.price_pre_final_projection
            # Minimum-distance mean-constraint projection is constant within month/peak cells.
            assert delta.groupby([months, peak]).std().fillna(0).max() < 1e-9
            gate_input = hourly[['price_shape']].copy()
            gate_input['ts_ch'] = local
            for field in ['year', 'month', 'quarter']:
                gate_input[field] = getattr(local, field)
            gates = build_product_normalization_gates(gate_input, surface,
                forward_date=pd.Timestamp(surface.date.iloc[0]), price_column='price_shape', hard_tolerance=1e-6, peak_country='CH')
            assert not gates.status.eq('CRITICAL').any()
            if ridge is None:
                old = pd.read_parquet(PRIOR/label/'signed-equal/curve.parquet')
                np.testing.assert_allclose(frame.price_shape, old.price_shape, atol=1e-9, rtol=0)
            if label == 'current':
                for grain, values in [('1h', hourly.price_shape), ('15min', frame.price_shape)]:
                    exported = pd.read_csv(dest/f'pfc-experimental-{grain}.csv', sep=';')
                    assert pd.DatetimeIndex(pd.to_datetime(exported.timestamp_utc, utc=True)).equals(values.index)
                    np.testing.assert_allclose(exported.price_eur_mwh, values, atol=6e-11, rtol=0)
            else:
                eligible = []
                for month in sorted(set(months)):
                    idx = hourly.index[months == month]
                    start = pd.Timestamp(month+'-01', tz='Europe/Zurich')
                    end = (pd.Period(month, freq='M')+1).start_time.tz_localize('Europe/Zurich')
                    expected_index = pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC')
                    if end <= pd.Timestamp('2026-09-01', tz='Europe/Zurich') and idx.equals(expected_index) and truth.reindex(idx).notna().all():
                        eligible.extend(idx)
                idx = pd.DatetimeIndex(eligible)
                if population is None:
                    population = idx
                assert population.equals(idx)
                for stage, column in [('final', 'price_shape'), ('pre_projection', 'price_pre_final_projection')]:
                    diagnostic = pd.read_parquet(dest/f'diagnostics-{stage}.parquet')
                    assert diagnostic.index.equals(idx)
                    actual, pred = truth.reindex(idx), hourly[column].reindex(idx)
                    month = idx.tz_convert('Europe/Zurich').strftime('%Y-%m')
                    full = pred-actual
                    le = full.groupby(month).transform('mean')
                    ramp, truth_ramp = full.diff(), actual.diff()
                    gaps = np.r_[True, (month[1:] != month[:-1]) | (np.diff(idx.as_unit('ns').asi8) != 3600000000000)]
                    ramp.iloc[gaps] = np.nan
                    truth_ramp.iloc[gaps] = np.nan
                    fields = dict(full_error=full, shape_error=full-le, level_error=le, ramp_error=ramp)
                    np.testing.assert_allclose(np.mean(full**2), np.mean((full-le)**2)+np.mean(le**2), atol=1e-8, rtol=0)
                    masks = independent_masks(idx, actual.to_numpy(), truth_ramp.to_numpy(), origin, plan['thresholds'][label])
                    signed = actual-actual.groupby(month).transform('mean')
                    masks.update(SHAPE_LOW=signed.to_numpy() < plan['thresholds'][label]['shape_p05'],
                        SHAPE_HIGH=signed.to_numpy() > plan['thresholds'][label]['shape_p95'],
                        SHAPE_ABS_TAIL=signed.abs().to_numpy() > plan['thresholds'][label]['shape_abs_p95'])
                    for field, values in fields.items():
                        np.testing.assert_allclose(diagnostic[field], values, atol=1e-9, rtol=0, equal_nan=True)
                        for segment, mask in masks.items():
                            v = values.loc[mask].dropna().to_numpy()
                            mae, rmse = (np.abs(v).mean(), np.sqrt(np.mean(v*v))) if len(v) else (np.nan, np.nan)
                            row = saved.loc[(label, candidate, stage, segment, field)]
                            assert row.hours == len(v)
                            np.testing.assert_allclose([row.mae, row.rmse], [mae, rmse], atol=1e-9, rtol=0, equal_nan=True)
                            metrics_count += 1
                            comparisons.append(dict(origin=label, role=spec['role'], candidate=candidate, stage=stage,
                                segment=segment, error_type=field, hours=len(v), mae=mae, rmse=rmse))
            replays.append(dict(origin=label, candidate=candidate, hours=len(hourly), product_gates=gates.status.value_counts().to_dict()))
        print(json.dumps(dict(stage='independent_origin_verified', origin=label)), flush=True)
    verified = pd.DataFrame(comparisons)
    decisions = json.loads((result/'decision.json').read_text())
    for decision in decisions['selections']:
        sub = verified.loc[verified.role.eq('assessment') & verified.stage.eq('final')]
        ref = sub.loc[sub.candidate.eq('signed-equal')].set_index(['origin', 'segment', 'error_type'])
        cand = sub.loc[sub.candidate.eq(decision['candidate'])].set_index(['origin', 'segment', 'error_type'])
        assert cand.index.equals(ref.index) and cand.hours.equals(ref.hours)
        adverse = []
        for key, row in cand.iterrows():
            if key[2] not in ['shape_error', 'ramp_error'] or row.hours == 0:
                continue
            r = ref.loc[key]
            if row.mae > 1.05*r.mae or row.rmse > 1.05*r.rmse:
                adverse.append(key)
        observed = [(r['origin'], r['segment'], r['error_type']) for r in decision['origin_regressions']]
        assert set(adverse) == set(observed)
        support = cand.reset_index().groupby(['segment', 'error_type']).agg(
            hours=('hours', 'sum'), origins=('hours', lambda x: int((x > 0).sum())))
        unsupported = [k for k in adverse if support.loc[k[1:]].hours < 168 or support.loc[k[1:]].origins < 2]
        assert len(unsupported) == decision['unsupported_adverse_count']
        assert set(unsupported) == {(r['origin'], r['segment'], r['error_type']) for r in decision['unsupported_adverse_segments']}
        allmask = (cand.index.get_level_values('segment') == 'ALL') & (cand.index.get_level_values('error_type') == 'shape_error')
        wins = int((cand.loc[allmask, 'mae'] < ref.loc[allmask, 'mae']).sum())
        assert wins == decision['wins']
        for metric in ['mae', 'rmse']:
            mask = (cand.index.get_level_values('segment') == 'ALL') & (cand.index.get_level_values('error_type') == 'shape_error')
            gain = 1-cand.loc[mask, metric].mean()/ref.loc[mask, metric].mean()
            assert abs(gain-decision['gains'][metric]) < 1e-12
        assert decision['adoption'] is False
        assert decision['local_screen_pass'] == bool(min(decision['gains'].values()) >= .02
            and wins >= 3 and not adverse and decision['shape_tail_support'])
    assert not any(plan['authority'].values()) and not any(decisions['authority'].values())
    write_json(out/'verification.json', dict(status='VERIFIED_LOCAL_INDEPENDENT_ARITHMETIC',
        metrics=metrics_count, replays=replays, max_solver_monthly_error=max_level,
        max_beta_error=max_beta, max_raw_error=max_raw, input_bindings=len(plan['inputs_sha256']),
        authority=dict(plan['authority'])))
    write_json(out/'manifest.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()})


if __name__ == '__main__':
    main()
