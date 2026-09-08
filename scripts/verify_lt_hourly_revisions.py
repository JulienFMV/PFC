"""Independent array reconstruction of D309 audit evidence, no fitting/adoption."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from pfc_shaping.validation.product_normalization import build_product_normalization_gates
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, sha, write_json
from scripts.verify_lt_hourly_recency import independent_prediction

PRIOR=ROOT/'build/lt-hourly-stability-20260908/run-v1'


def array_components(curve,levels):
    price=curve.price_shape.resample('h').mean()
    pre=curve.price_pre_final_projection.resample('h').mean()
    index=price.index
    base=np.array([levels[key] for key in index.tz_convert('Europe/Zurich').strftime('%Y-%m')])
    return index,np.column_stack([price,base,pre.to_numpy()-base,price.to_numpy()-pre.to_numpy()])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run,out=args.run.resolve(),args.output.resolve()
    task=ROOT/'build/lt-hourly-revision-audit-20260908'
    if Path.cwd()!=ROOT or not run.is_relative_to(task) or not out.is_relative_to(task) or out==task:
        raise ValueError('canonical audit paths required')
    out.mkdir(exist_ok=False)
    plan=json.loads((run/'plan.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    complete=json.loads((run/'complete.json').read_text())
    for name,digest in manifest.items(): assert sha(run/name)==digest,name
    for name,digest in {**plan['inputs_sha256'],**plan['code_sha256']}.items(): assert sha(ROOT/name)==digest,name
    assert len(plan['authority'])==6 and not any(plan['authority'].values()) and not any(complete['authority'].values())
    assert plan['source_policy']=='LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT'
    assert plan['real_daily_D304_vintage_status']=='UNSUPPORTED_NO_ADMITTED_ARCHIVED_SERIES'
    events=pd.read_parquet(run/'events.parquet')
    csv=pd.read_csv(run/'events.csv',dtype={'origin':str})
    assert pd.DatetimeIndex(pd.to_datetime(csv.timestamp_utc,utc=True)).equals(events.index)
    for field in ['final_step','level_step','assembly_shape_step','projection_step','raw_step','centering_step',
                  'full_ramp_error','shape_ramp_error','level_ramp_error']:
        np.testing.assert_allclose(csv[field],events[field],atol=1e-9,rtol=0,equal_nan=True)
    checked_events=0
    arrays={}
    solvers={}
    max_identity=max_solver=0.
    for spec in plan['origins']:
        label=spec['id']; origin=pd.Timestamp(spec['origin_utc'])
        target=pd.read_parquet(PRIOR/label/'targets.parquet')
        assert target.index[-1]<origin
        local=target.index.tz_convert('Europe/Zurich')
        months=local.strftime('%Y-%m').to_numpy()
        for m in set(months):
            start=pd.Timestamp(m+'-01',tz='Europe/Zurich')
            end=(pd.Period(m,freq='M')+1).start_time.tz_localize('Europe/Zurich')
            assert end<=origin and target.loc[months==m].index.equals(pd.date_range(start,end,freq='h',inclusive='left').tz_convert('UTC'))
        valid=np.r_[False,np.diff(target.index.as_unit('ns').asi8)==3_600_000_000_000]
        boundary=valid&np.r_[False,months[1:]!=months[:-1]]
        midnight=valid&(local.hour==0)&~boundary
        ramp=np.r_[np.nan,np.diff(target.price_eur_mwh)]
        signed=np.r_[np.nan,np.diff(target.signed_target)]
        scale=plan['scales'][label]
        assert scale['boundary_count']==int(boundary.sum()) and scale['midnight_count']==int(midnight.sum())
        for key,values in [('boundary_p95',np.abs(ramp[boundary])),('signed_boundary_p95',np.abs(signed[boundary])),('midnight_p95',np.abs(ramp[midnight]))]:
            if key!='midnight_p95' and boundary.sum()<12: assert scale[key] is None
            else: np.testing.assert_allclose(scale[key],np.quantile(values,.95),atol=1e-12,rtol=0)
        ordered=sorted(set(months))
        if len(ordered)>=24:
            older=target.loc[months<ordered[-12]]
            assert scale['older_training_end']==older.index[-1].isoformat()
            first=origin.tz_convert('Europe/Zurich').tz_localize(None).to_period('M')+1
            grid=pd.date_range(first.start_time,(first+12).start_time,freq='h',inclusive='left',tz='Europe/Zurich').tz_convert('UTC')
            a,_=independent_prediction(target.signed_target,pd.Series(1.,index=target.index),grid)
            b,_=independent_prediction(older.signed_target,pd.Series(1.,index=older.index),grid)
            delta=a-b
            probe_month=grid.tz_convert('Europe/Zurich').strftime('%Y-%m')
            for m in set(probe_month): delta[probe_month==m]-=np.mean(delta[probe_month==m])
            np.testing.assert_allclose(scale['revision_p95'],np.quantile(np.abs(delta),.95),atol=1e-10,rtol=0)
            assert scale['revision_probe_hours']==len(grid)
        else: assert scale['revision_p95'] is None and scale['revision_status']=='UNSUPPORTED'
        solver=json.loads((SOURCE/'monthly-solver/result.json' if label=='current' else OLD/f'{label}/solver.json').read_text())
        levels=solver['assembler_base_prices'] if label=='current' else solver['base_prices']
        solvers[label]=levels
        actual_series=(pd.Series(dtype=float,index=pd.DatetimeIndex([],tz='UTC')) if label=='current' else
                       pd.read_parquet(PRIOR/label/'signed-equal/diagnostics-final.parquet').actual)
        for candidate in plan['candidates']:
            curve=pd.read_parquet(PRIOR/label/candidate/'curve.parquet')
            index,parts=array_components(curve,levels)
            arrays[(label,candidate)]=(index,parts)
            local=index.tz_convert('Europe/Zurich')
            month=local.strftime('%Y-%m').to_numpy()
            season=enrich_15min_index(index).saison.to_numpy()
            valid=np.r_[False,np.diff(index.as_unit('ns').asi8)==3_600_000_000_000]
            mb=valid&np.r_[False,month[1:]!=month[:-1]]
            sb=valid&np.r_[False,season[1:]!=season[:-1]]
            md=valid&(local.hour==0)
            use=md|mb|sb
            saved=events.loc[events.origin.eq(label)&events.candidate.eq(candidate)]
            assert saved.index.equals(index[use])
            for field,val in [('month_boundary',mb),('season_boundary',sb),('midnight',md),('contiguous',valid)]:
                np.testing.assert_array_equal(saved[field],val[use])
            diffs=np.vstack([np.full((1,4),np.nan),np.diff(parts,axis=0)])
            diffs[~valid]=np.nan
            for j,field in enumerate(['final','level','assembly_shape','projection']):
                np.testing.assert_allclose(saved[field+'_step'],diffs[use,j],atol=1e-10,rtol=0)
            np.testing.assert_allclose(diffs[use,0],diffs[use,1:].sum(axis=1),atol=1e-9,rtol=0)
            if candidate!='mlp-control':
                raw=pd.read_parquet(PRIOR/label/candidate/'raw.parquet').raw.to_numpy()
                rd=np.r_[np.nan,np.diff(raw)]
                cd=np.r_[np.nan,np.diff(parts[:,2]-raw)]
                np.testing.assert_allclose(saved.raw_step,rd[use],atol=1e-10,rtol=0)
                np.testing.assert_allclose(saved.centering_step,cd[use],atol=1e-10,rtol=0)
            truth=actual_series.reindex(index).to_numpy()
            means=np.full(len(index),np.nan)
            for m in set(month):
                where=month==m
                if np.isfinite(truth[where]).all(): means[where]=np.mean(truth[where])
                observed=float(np.mean(parts[where,0]))
                drift=abs(observed-levels[m]); max_solver=max(max_solver,drift)
                assert drift<1e-9
            actual_diff=np.r_[np.nan,np.diff(truth)]
            full=diffs[:,0]-actual_diff
            level=np.r_[np.nan,np.diff(parts[:,1]-means)]
            shape=np.r_[np.nan,np.diff(parts[:,0]-parts[:,1]-truth+means)]
            for field,value in [('actual',truth),('truth_step',actual_diff),('full_ramp_error',full),('level_ramp_error',level),('shape_ramp_error',shape)]:
                np.testing.assert_allclose(saved[field],value[use],atol=1e-9,rtol=0,equal_nan=True)
            scored=valid&np.isfinite(truth)&np.r_[False,np.isfinite(truth[:-1])]
            np.testing.assert_array_equal(saved.scored,scored[use])
            np.testing.assert_allclose(full[scored],level[scored]+shape[scored],atol=1e-9,rtol=0)
            if scored.any(): max_identity=max(max_identity,float(np.max(np.abs(full[scored]-level[scored]-shape[scored]))))
            threshold=np.where(mb,scale['boundary_p95'],scale['midnight_p95']).astype(float)
            np.testing.assert_allclose(saved.threshold,threshold[use],equal_nan=True,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(saved.forecast_step_exceeds_scale,np.abs(diffs[use,0])>threshold[use])
            np.testing.assert_array_equal(saved.delivery_year,local.year[use])
            np.testing.assert_array_equal(saved.season,season[use])
            lead=local.year*12+local.month-(origin.tz_convert('Europe/Zurich').year*12+origin.tz_convert('Europe/Zurich').month)
            np.testing.assert_array_equal(saved.lead_months,lead[use])
            horizon=np.select([lead<=6,lead<=12,lead<=24,lead<=36],['M01_M06','M07_M12','M13_M24','M25_M36'],default='M37_PLUS')
            np.testing.assert_array_equal(saved.horizon,horizon[use])
            for field,mask in [('negative_truth',truth<0),('near_zero_truth',np.abs(truth)<=5),('positive_truth',truth>5),
                               ('shape_tail_truth',np.abs(truth-means)>plan['tail_thresholds'][label]['shape_abs_p95']),
                               ('high_price_truth',truth>plan['tail_thresholds'][label]['price_p95'])]:
                np.testing.assert_array_equal(saved[field],mask[use])
            checked_events+=len(saved)
        print(json.dumps(dict(stage='verified_events',origin=label)),flush=True)
    assert checked_events==len(events)
    metrics=pd.read_csv(run/'seam-metrics.csv',dtype={'origin':str})
    event_groups={(str(origin),candidate):part for (origin,candidate),part in events.groupby(['origin','candidate'])}
    for row in metrics.itertuples():
        part=event_groups[(row.origin,row.candidate)]
        if row.kind=='MONTH_BOUNDARY': part=part.loc[part.month_boundary]
        elif row.kind=='SEASON_BOUNDARY': part=part.loc[part.season_boundary]
        elif row.kind=='OTHER_MONTH_BOUNDARY': part=part.loc[part.month_boundary&~part.season_boundary]
        else: assert row.kind=='WITHIN_MONTH_MIDNIGHT'; part=part.loc[part.midnight&~part.month_boundary]
        if ':' in row.segment:
            key,val=row.segment.split(':'); part=part.loc[part[key].astype(str).eq(val)]
        elif row.segment!='ALL': part=part.loc[part[row.segment]]
        values=part.loc[part.scored,row.error_type].dropna().to_numpy()
        assert row.events==len(part) and row.scored_events==len(values)
        expected=[np.mean(np.abs(values)),np.sqrt(np.mean(values**2)),np.mean(values),np.quantile(np.abs(values),.95)] if len(values) else [np.nan]*4
        np.testing.assert_allclose([row.mae,row.rmse,row.bias,row.p95],expected,atol=1e-9,rtol=0,equal_nan=True)
        assert row.status==('SCORED_RETROSPECTIVE' if len(values) else 'UNSUPPORTED')
    pair_status=json.loads((run/'pair-status.json').read_text())
    revision_files=0; counterfactuals=0; attribution_rows=0
    revision_cache={}
    for pair in pair_status:
        name=pair['pair']; a,b=name.split('-to-')
        ia,va=arrays[(a,'signed-equal')]; ib,vb=arrays[(b,'signed-equal')]
        common=ia.intersection(ib)
        assert pair['hours']==len(common)
        if not len(common): assert pair['status']=='UNSUPPORTED_NO_COMMON_DELIVERY'; continue
        for candidate in plan['candidates']:
            idx0,values0=arrays[(a,candidate)]; idx1,values1=arrays[(b,candidate)]
            delta=values1[idx1.get_indexer(common)]-values0[idx0.get_indexer(common)]
            saved=pd.read_parquet(run/'revisions'/f'{name}-{candidate}.parquet')
            assert saved.index.equals(common)
            np.testing.assert_allclose(saved,delta,atol=1e-9,rtol=0)
            revision_cache[(name,candidate)]=saved
            revision_files+=1
        folder=run/'counterfactuals'/name
        curve=pd.read_parquet(folder/'curve.parquet')
        index,cf=array_components(curve,solvers[b])
        target=pd.read_parquet(PRIOR/a/'targets.parquet')
        raw,_=independent_prediction(target.signed_target,pd.Series(1.,index=target.index),index)
        np.testing.assert_allclose(pd.read_parquet(folder/'raw.parquet').raw,raw,atol=1e-10,rtol=0)
        centered=raw.copy(); months=index.tz_convert('Europe/Zurich').strftime('%Y-%m')
        for m in set(months): centered[months==m]-=np.mean(centered[months==m])
        np.testing.assert_allclose(cf[:,2],centered,atol=1e-9,rtol=0)
        for m in set(months): assert abs(float(np.mean(cf[months==m,0]))-solvers[b][m])<1e-9
        before=va[ia.get_indexer(common)]; after=vb[ib.get_indexer(common)]; c=cf[index.get_indexer(common)]
        expected=np.column_stack([after[:,0]-before[:,0],c[:,1]-before[:,1],c[:,2]-before[:,2],c[:,3]-before[:,3],
                                  after[:,2]-c[:,2],after[:,3]-c[:,3],after[:,0]-c[:,0],c[:,0]-before[:,0]])
        saved=pd.read_parquet(folder/'attribution.parquet')
        assert saved.index.equals(common)
        np.testing.assert_allclose(saved,expected,atol=1e-9,rtol=0)
        np.testing.assert_allclose(expected[:,0],expected[:,1:6].sum(axis=1),atol=1e-9,rtol=0)
        np.testing.assert_allclose(expected[:,6],expected[:,4:6].sum(axis=1),atol=1e-9,rtol=0)
        h=pd.DataFrame(dict(price_eur_mwh=cf[:,0],ts_ch=index.tz_convert('Europe/Zurich')),index=index)
        for field in ['year','month','quarter']: h[field]=getattr(h.ts_ch.dt,field)
        surface=(select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet'))
                 if b=='current' else pd.read_parquet(OLD/f'{b}/eex-surface.parquet'))
        gates=build_product_normalization_gates(h,surface,forward_date=pd.Timestamp(surface.date.iloc[0]),price_column='price_eur_mwh',hard_tolerance=1e-6,peak_country='CH')
        old=pd.read_csv(folder/'product-gates.csv')
        for column in gates.select_dtypes(include='object'):
            gates[column]=gates[column].fillna(''); old[column]=old[column].fillna('')
        pd.testing.assert_frame_equal(gates.reset_index(drop=True),old,check_dtype=False,atol=1e-9,rtol=1e-10)
        assert not gates.status.eq('CRITICAL').any()
        if b=='current':
            assert gates.status.value_counts().to_dict()=={'PASS':80,'QUOTE_CONFLICT':9}
            main=curve.loc[curve.index<pd.Timestamp('2030-01-01',tz='Europe/Zurich')]
            for resolution,series in [('1h',main.price_shape.resample('h').mean()),('15min',main.price_shape)]:
                export=pd.read_csv(folder/f'counterfactual-ch-{resolution}.csv',sep=';')
                assert pd.DatetimeIndex(pd.to_datetime(export.timestamp_utc,utc=True)).equals(series.index)
                assert export.timestamp_ch.tolist()==series.index.tz_convert('Europe/Zurich').map(lambda t:t.isoformat()).tolist()
                np.testing.assert_allclose(export.price_eur_mwh,series,atol=5.1e-11,rtol=0)
        np.testing.assert_allclose(np.ptp(curve.price_shape.to_numpy().reshape(-1,4),axis=1),0,atol=1e-9,rtol=0)
        counterfactuals+=1; attribution_rows+=len(common)
    assert counterfactuals==complete['counterfactual_assemblies']
    summary=pd.read_csv(run/'revision-summary.csv',dtype={'delivery_year':str})
    for row in summary.itertuples():
        delta=revision_cache[(row.pair,row.candidate)]
        if row.delivery_year!='ALL': delta=delta.loc[delta.index.tz_convert('Europe/Zurich').year==int(row.delivery_year)]
        val=delta[row.component].to_numpy()
        assert row.hours==len(val)
        np.testing.assert_allclose([row.mean,row.mean_absolute,row.p95_absolute,row.maximum_absolute],
            [np.mean(val),np.mean(np.abs(val)),np.quantile(np.abs(val),.95),np.max(np.abs(val))],atol=1e-9,rtol=0)
        if row.component=='assembly_shape' and pd.notna(row.input_shape_scale): assert row.shape_exceedances==int((np.abs(val)>row.input_shape_scale).sum())
    rec=json.loads((run/'recommendation.json').read_text())
    d=events.loc[events.origin.isin(['2023','2024','2025','2026'])&events.candidate.eq('signed-equal')&events.scored]
    s=d.loc[d.season_boundary]; m=d.loc[d.month_boundary&~d.season_boundary]
    support=len(s)>=8 and len(m)>=8 and s.origin.nunique()>=3 and m.origin.nunique()>=3
    ratios=[np.mean(np.abs(s[field]))/np.mean(np.abs(m[field])) for field in ['full_ramp_error','shape_ramp_error']]
    share=np.abs(s.raw_step).sum()/(np.abs(s.raw_step).sum()+np.abs(s.centering_step).sum()+np.abs(s.projection_step).sum())
    assert rec['support']==bool(support) and rec['seasonal_events']==len(s) and rec['other_month_events']==len(m)
    np.testing.assert_allclose(list(rec['mae_ratios'].values()),ratios,atol=1e-12,rtol=0)
    np.testing.assert_allclose(rec['raw_absolute_contribution_share'],share,atol=1e-12,rtol=0)
    assert rec['smooth_calendar_hypothesis_warranted']==bool(support and min(ratios)>1.1 and share>.5)
    assert not rec['adoption'] and not any(rec['authority'].values())
    draft=json.loads((run/'prospective-holdout-draft.json').read_text())
    assert draft['status']=='DRAFT_NOT_INDEPENDENTLY_REGISTERED' and draft['independent_custodian'] is None and not any(draft['authority'].values())
    assert sha(ROOT/draft['forecast_path'])==draft['forecast_sha256']
    assert pd.Timestamp(draft['delivery_start'])>pd.Timestamp(plan['frozen_at_utc'])
    oldinventory=json.loads((OLD/'session-artifacts.json').read_text())
    for name,entry in oldinventory['files'].items(): assert sha(OLD/name)==entry['sha256'],name
    verification=dict(status='VERIFIED',inputs=len(plan['inputs_sha256']),sources=len(plan['code_sha256']),outputs=len(manifest),
        event_rows=checked_events,seam_metric_rows=len(metrics),revision_series=revision_files,revision_metric_rows=len(summary),
        counterfactuals=counterfactuals,attribution_hours=attribution_rows,max_solver_error=max_solver,
        max_boundary_identity_error=max_identity,unchanged_D301_files=len(oldinventory['files']),authority=plan['authority'])
    write_json(out/'verification.json',verification)
    selected=metrics.loc[metrics.role.eq('assessment')&metrics.segment.eq('ALL')]
    selected.to_csv(out/'assessment-seams.csv',index=False)
    write_json(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(verification=verification,recommendation=rec)),flush=True)


if __name__=='__main__': main()
