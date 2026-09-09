"""Independent numerical replay of D310 pairs, ridge, scores and constraints."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from pfc_shaping.validation.product_normalization import build_product_normalization_gates
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, sha, write_json
from scripts.verify_lt_hourly_recency import independent_prediction, independent_masks
from scripts.verify_lt_hourly_stability import screen


def independent_basis(index, origin, maturity):
    cal = enrich_15min_index(index)
    local = index.tz_convert('Europe/Zurich')
    o = origin.tz_convert('Europe/Zurich')
    lead = np.asarray(12*(local.year-o.year)+local.month-o.month)
    vectors = []
    for field, values in [('saison', ['Hiver', 'Printemps', 'Ete', 'Automne']),
                          ('type_jour', ['Ouvrable', 'Samedi', 'Dimanche', 'Ferie_CH', 'Ferie_DE'])]:
        for value in values:
            for harmonic in [1, 2, 3]:
                angle = cal.heure_hce.to_numpy()*harmonic*np.pi/12
                vectors.extend([np.sin(angle)*(cal[field].to_numpy() == value),
                                np.cos(angle)*(cal[field].to_numpy() == value)])
    x = np.array(vectors).T
    if maturity:
        x = np.concatenate([x, x*np.clip(lead, 0, 36)[:, None]/36], axis=1)
    months = local.strftime('%Y-%m')
    for month in np.unique(months):
        mask = months == month
        x[mask] -= x[mask].mean(axis=0)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    task = ROOT/'build/lt-maturity-20260908'
    prior = ROOT/'build/lt-hourly-stability-20260908/run-v1'
    if Path.cwd() != ROOT or not run.is_relative_to(task) or not out.is_relative_to(task) or out == task:
        raise ValueError('canonical task paths required')
    out.mkdir(exist_ok=False)
    plan = json.loads((run/'plan.json').read_text())
    complete = json.loads((run/'complete.json').read_text())
    for name, digest in json.loads((run/'manifest.json').read_text()).items():
        assert sha(run/name) == digest, name
    for name, digest in {**plan['inputs_sha256'], **plan['code_sha256']}.items():
        assert sha(ROOT/name) == digest, name
    assert not any(plan['authority'].values()) and not any(complete['authority'].values())
    truth_qh = pd.read_parquet(SOURCE/'prepared-inputs/epex-ch.parquet').price_eur_mwh
    assert (truth_qh.resample('h').max()-truth_qh.resample('h').min()).eq(0).all()
    truth = truth_qh.resample('h').mean()
    saved = pd.read_csv(run/'diagnostic-metrics.csv', dtype={'origin': str}).set_index(['origin', 'candidate', 'stage', 'segment', 'error_type'])
    replay_rows, verified_metrics = [], []
    max_level = 0.
    older = {}
    events = pd.read_parquet(run/'events.parquet')
    for spec in plan['origins']:
        label, origin = spec['id'], pd.Timestamp(spec['origin_utc'])
        target = pd.read_parquet(prior/label/'targets.parquet')
        pair = pd.read_parquet(run/label/'pairs.parquet')
        assert (pair.delivery < origin).all() and (pair.history_end < pair.origin).all()
        assert not pair.duplicated(['origin', 'delivery']).any()
        np.testing.assert_allclose(pair.groupby('delivery').weight.sum(), 1, atol=1e-14)
        matrices = {False: [np.zeros((54, 54)), np.zeros(54)], True: [np.zeros((108, 108)), np.zeros(108)]}
        for inner, part in pair.groupby('origin'):
            index = pd.DatetimeIndex(part.delivery)
            month = index.tz_convert('Europe/Zurich').strftime('%Y-%m')
            history = target.loc[target.index.tz_convert('Europe/Zurich').strftime('%Y-%m') < inner.tz_convert('Europe/Zurich').strftime('%Y-%m')]
            assert history.index[-1] == part.history_end.iloc[0]
            assert len(set(history.index.tz_convert('Europe/Zurich').strftime('%Y-%m'))) >= 12
            forecast, _ = independent_prediction(history.signed_target, pd.Series(1., index=history.index), index)
            forecast = pd.Series(forecast, index=index)
            centered = forecast-forecast.groupby(month).transform('mean')
            np.testing.assert_allclose(part.baseline, centered, atol=1e-10, rtol=0)
            np.testing.assert_allclose(part.residual, target.signed_target.reindex(index).to_numpy()-centered.to_numpy(), atol=1e-10, rtol=0)
            for maturity in [False, True]:
                x = independent_basis(index, inner, maturity)
                wx = x*np.sqrt(part.weight.to_numpy())[:, None]
                wy = part.residual.to_numpy()*np.sqrt(part.weight.to_numpy())
                matrices[maturity][0] += wx.T@wx
                matrices[maturity][1] += wx.T@wy
        solver = json.loads((SOURCE/'monthly-solver/result.json' if label == 'current' else OLD/f'{label}/solver.json').read_text())
        levels = solver['assembler_base_prices'] if label == 'current' else solver['base_prices']
        surface = (select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet'))
                   if label == 'current' else pd.read_parquet(OLD/f'{label}/eex-surface.parquet'))
        populations = {}
        for candidate in plan['candidates']:
            dest = run/label/candidate
            curve = pd.read_parquet(dest/'curve.parquet')
            raw = pd.read_parquet(dest/'raw.parquet').raw
            reference = pd.read_parquet(prior/label/'signed-equal/raw.parquet').raw
            if candidate == 'signed-equal':
                assert sha(dest/'curve.parquet') == sha(prior/label/candidate/'curve.parquet')
            else:
                maturity = candidate == 'residual-maturity'
                gram, rhs = matrices[maturity]
                values, vectors = np.linalg.eigh(gram/pair.weight.sum())
                beta = vectors@((vectors.T@(rhs/pair.weight.sum()))/(values+.1))
                saved_beta = json.loads((dest/'fit.json').read_text())['coefficients']
                np.testing.assert_allclose(beta, saved_beta, atol=1e-9, rtol=0)
                expected = reference.to_numpy()+independent_basis(raw.index, origin, maturity)@beta
                np.testing.assert_allclose(raw, expected, atol=1e-9, rtol=0)
            h = curve[['price_shape', 'price_pre_final_projection']].resample('h').mean()
            month = h.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
            level = np.array([levels[m] for m in month])
            for column in h:
                means = h[column].groupby(month).mean()
                discrepancy = max(abs(means[m]-levels[m]) for m in means.index)
                max_level = max(max_level, discrepancy)
                assert discrepancy < 1e-9
            gate_input = h[['price_shape']].copy()
            gate_input['ts_ch'] = h.index.tz_convert('Europe/Zurich')
            gate_input['year'] = gate_input.ts_ch.dt.year
            gate_input['month'] = gate_input.ts_ch.dt.month
            gate_input['quarter'] = gate_input.ts_ch.dt.quarter
            gates = build_product_normalization_gates(gate_input, surface,
                forward_date=pd.Timestamp(surface.date.iloc[0]), price_column='price_shape',
                hard_tolerance=1e-6, peak_country='CH')
            assert not gates.status.eq('CRITICAL').any()
            for stage, column in [('final', 'price_shape'), ('pre_projection', 'price_pre_final_projection')]:
                if label == 'current':
                    continue
                diagnostic = pd.read_parquet(dest/f'diagnostics-{stage}.parquet')
                idx = diagnostic.index
                from pfc_shaping.lt.benchmark_safeguards import require_common_population
                require_common_population(populations.setdefault(stage, idx), idx, origin)
                pred, actual = h[column].reindex(idx), truth.reindex(idx)
                group = idx.tz_convert('Europe/Zurich').strftime('%Y-%m')
                le = pred.groupby(group).transform('mean')-actual.groupby(group).transform('mean')
                full = pred-actual
                ramp = full.diff()
                ramp.iloc[np.r_[True, (group[1:] != group[:-1]) | (np.diff(idx.as_unit('ns').asi8) != 3600000000000)]] = np.nan
                fields = dict(shape_error=full-le, full_error=full, level_error=le, ramp_error=ramp)
                truth_ramp = actual.diff()
                truth_ramp.iloc[np.r_[True, (group[1:] != group[:-1]) | (np.diff(idx.as_unit('ns').asi8) != 3600000000000)]] = np.nan
                masks = independent_masks(idx, actual.to_numpy(), truth_ramp.to_numpy(), origin, plan['thresholds'][label])
                signed = actual-actual.groupby(group).transform('mean')
                masks.update(SHAPE_LOW=signed.to_numpy() < plan['thresholds'][label]['shape_p05'],
                    SHAPE_HIGH=signed.to_numpy() > plan['thresholds'][label]['shape_p95'],
                    SHAPE_ABS_TAIL=signed.abs().to_numpy() > plan['thresholds'][label]['shape_abs_p95'])
                for field, values in fields.items():
                    np.testing.assert_allclose(diagnostic[field], values, atol=1e-9, rtol=0, equal_nan=True)
                    for segment, mask in masks.items():
                        v = values.loc[mask].dropna().to_numpy()
                        mae, rmse = (float(np.abs(v).mean()), float(np.sqrt(np.mean(v*v)))) if len(v) else (np.nan, np.nan)
                        record = saved.loc[(label, candidate, stage, segment, field)]
                        assert record.hours == len(v)
                        np.testing.assert_allclose([record.mae, record.rmse], [mae, rmse], atol=1e-9, rtol=0, equal_nan=True)
                        verified_metrics.append(dict(origin=label, role=spec['role'], candidate=candidate, stage=stage,
                            segment=segment, error_type=field, hours=len(v), mae=mae, rmse=rmse))
            e = events.loc[events.origin.eq(label) & events.candidate.eq(candidate)]
            steps = h.price_shape.diff().reindex(e.index)
            np.testing.assert_allclose(e.final_step, steps, atol=1e-9, rtol=0)
            np.testing.assert_allclose(e.final_step, e.level_step+e.assembly_shape_step+e.projection_step, atol=1e-9, rtol=0)
            np.testing.assert_allclose(e.full_ramp_error, e.shape_ramp_error+e.level_ramp_error, atol=1e-9, rtol=0, equal_nan=True)
            parts = pd.DataFrame(dict(final=h.price_shape, level=level, assembly_shape=h.price_pre_final_projection-level,
                                      projection=h.price_shape-h.price_pre_final_projection), index=h.index)
            if candidate in older:
                common = parts.index.intersection(older[candidate].index)
                expected = parts.loc[common]-older[candidate].loc[common]
                pd.testing.assert_frame_equal(pd.read_parquet(dest/'revision-from-previous.parquet'), expected, atol=1e-9, rtol=0)
            older[candidate] = parts
            replay_rows.append(dict(origin=label, candidate=candidate, hours=len(h), gates=gates.status.value_counts().to_dict()))
        print(json.dumps(dict(stage='independent_origin_verified', origin=label)), flush=True)
    verified = pd.DataFrame(verified_metrics)
    assessment = verified.loc[verified.role.eq('assessment') & verified.stage.eq('final')]
    comparison = assessment.groupby(['candidate', 'segment', 'error_type'], as_index=False).agg(
        hours=('hours', 'sum'), origins=('mae', 'count'), mae=('mae', 'mean'), rmse=('rmse', 'mean'))
    gateframe, originframe, _, selections = screen(comparison, assessment, plan['candidates'][1:])
    seam_rows = []
    pooled = events.loc[events.role.eq('assessment') & events.scored & events.month_boundary]
    for choice in selections:
        candidate = choice['candidate']
        seam_pass = True
        for field in ['shape_ramp_error', 'full_ramp_error']:
            part = pooled.loc[pooled.candidate.eq(candidate)]
            ref = pooled.loc[pooled.candidate.eq('signed-equal')]
            supported = len(part) >= 8 and part.origin.nunique() >= 3
            a = float(part[field].abs().mean()/ref[field].abs().mean())
            b = float(np.sqrt(np.mean(part[field]**2)/np.mean(ref[field]**2)))
            passed = bool(supported and max(a, b) <= 1.05)
            seam_pass &= passed
            seam_rows.append(dict(candidate=candidate, error_type=field, events=len(part), mae_ratio=a, rmse_ratio=b, passed=passed))
        choice['seam_screen_pass'] = seam_pass
        choice['combined_screen_pass'] = choice['local_screen_pass'] and seam_pass
    allshape = comparison.loc[comparison.segment.eq('ALL') & comparison.error_type.eq('shape_error')].set_index('candidate')
    maturity_gain = {key: float(100*(1-allshape.loc['residual-maturity', key]/allshape.loc['residual-calendar', key])) for key in ['mae', 'rmse']}
    comparison.to_csv(out/'comparison.csv', index=False)
    gateframe.to_csv(out/'regime-gates.csv', index=False)
    originframe.to_csv(out/'origin-gates.csv', index=False)
    pd.DataFrame(seam_rows).to_csv(out/'seam-gates.csv', index=False)
    write_json(out/'decision.json', dict(selections=selections, maturity_gain_vs_calendar_pct=maturity_gain,
        retained_reference='D304', authority=plan['authority']))
    write_json(out/'verification.json', dict(status='VERIFIED', replays=replay_rows, metrics=len(verified_metrics),
        events=len(events), max_solver_monthly_error=max_level, authority=plan['authority']))
    write_json(out/'manifest.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()})


if __name__ == '__main__':
    main()
