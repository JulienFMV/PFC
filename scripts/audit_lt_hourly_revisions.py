"""D309 local revision/seam audit; no historical-vintage or model authority."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape
from scripts.run_lt_hourly_stability import training_thresholds
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, assembler, save_curve, sha, write_json

WORK=ROOT/'build/lt-hourly-revision-audit-20260908'
PRIOR=ROOT/'build/lt-hourly-stability-20260908/run-v1'
ERRORS=('full_ramp_error','shape_ramp_error','level_ramp_error')


def transitions(index):
    if not isinstance(index,pd.DatetimeIndex) or index.tz is None or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('ordered unique timezone-aware hourly index required')
    local=index.tz_convert('Europe/Zurich')
    contiguous=np.r_[False,np.diff(index.as_unit('ns').asi8)==3_600_000_000_000]
    month=local.strftime('%Y-%m').to_numpy()
    season=enrich_15min_index(index).saison.to_numpy()
    return pd.DataFrame(dict(contiguous=contiguous,midnight=contiguous&(local.hour==0),
        month_boundary=contiguous&np.r_[False,month[1:]!=month[:-1]],
        season_boundary=contiguous&np.r_[False,season[1:]!=season[:-1]]),index=index)


def preorigin_scales(target,origin):
    # Reuse the established closed-month, finite-value and cutoff contract.
    training_thresholds(target,origin)
    flags=transitions(target.index)
    ramp=target.price_eur_mwh.diff().abs()
    boundary=ramp.loc[flags.month_boundary]
    midnight=ramp.loc[flags.midnight&~flags.month_boundary]
    signed=target.signed_target.diff().abs().loc[flags.month_boundary]
    result=dict(boundary_count=len(boundary),midnight_count=len(midnight),
        boundary_p95=float(boundary.quantile(.95)) if len(boundary)>=12 else None,
        midnight_p95=float(midnight.quantile(.95)) if len(midnight) else None,
        signed_boundary_p95=float(signed.quantile(.95)) if len(boundary)>=12 else None,
        revision_p95=None,revision_status='UNSUPPORTED',revision_probe_hours=0)
    months=target.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    ordered=sorted(set(months))
    if len(ordered)>=24:
        older=target.loc[months<ordered[-12]]
        start=origin.tz_convert('Europe/Zurich').tz_localize(None).to_period('M')+1
        grid=pd.date_range(start.start_time,(start+12).start_time,freq='h',inclusive='left',tz='Europe/Zurich').tz_convert('UTC')
        all_raw,_=calendar_cell_reference(target.signed_target,grid)
        old_raw,_=calendar_cell_reference(older.signed_target,grid)
        delta=center_signed_hourly_shape(pd.Series(all_raw-old_raw,index=grid))
        result.update(revision_p95=float(delta.abs().quantile(.95)),revision_status='PREORIGIN_PSEUDO_UPDATE_SCALE',
                      revision_probe_hours=len(grid),older_training_end=older.index[-1].isoformat())
    return result


def components(curve,levels):
    h=curve[['price_shape','price_pre_final_projection']].resample('h').mean()
    flags=transitions(h.index)
    if not flags.contiguous.iloc[1:].all() or not np.isfinite(h).all().all():
        raise ValueError('complete finite hourly curve required')
    level=pd.Series(h.index.tz_convert('Europe/Zurich').strftime('%Y-%m').map(levels),index=h.index,dtype=float)
    if not np.isfinite(level).all():
        raise ValueError('every month requires a solver level')
    return pd.DataFrame(dict(final=h.price_shape,level=level,
        assembly_shape=h.price_pre_final_projection-level,
        projection=h.price_shape-h.price_pre_final_projection))


def event_frame(parts,raw,actual,origin,scales,tail_thresholds):
    index=parts.index
    flags=transitions(index)
    month=index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    actual=actual.reindex(index)
    actual_level=actual.groupby(month).transform('mean')
    steps=parts.diff().where(flags.contiguous,axis=0)
    truth_step=actual.diff().where(flags.contiguous)
    frame=flags.copy()
    for column in steps:
        frame[column+'_step']=steps[column]
    frame['actual']=actual
    frame['truth_step']=truth_step
    frame['scored']=actual.notna()&actual.shift().notna()&flags.contiguous
    frame['full_ramp_error']=steps.final-truth_step
    frame['level_ramp_error']=(parts.level-actual_level).diff().where(flags.contiguous)
    frame['shape_ramp_error']=((parts.final-parts.level)-(actual-actual_level)).diff().where(flags.contiguous)
    if raw is not None:
        if not raw.index.equals(index): raise ValueError('raw prediction index mismatch')
        frame['raw_step']=raw.diff().where(flags.contiguous)
        frame['centering_step']=(parts.assembly_shape-raw).diff().where(flags.contiguous)
    else:
        frame['raw_step']=np.nan
        frame['centering_step']=np.nan
    frame['threshold']=np.where(flags.month_boundary,scales['boundary_p95'],scales['midnight_p95'])
    frame['scale_supported']=pd.notna(frame.threshold)
    frame['forecast_step_exceeds_scale']=frame.final_step.abs()>pd.to_numeric(frame.threshold)
    local=index.tz_convert('Europe/Zurich')
    frame['delivery_year']=local.year
    frame['season']=enrich_15min_index(index).saison.to_numpy()
    leads=local.year*12+local.month-(origin.tz_convert('Europe/Zurich').year*12+origin.tz_convert('Europe/Zurich').month)
    frame['lead_months']=leads
    frame['horizon']=pd.cut(leads,bins=[0,6,12,24,36,np.inf],labels=['M01_M06','M07_M12','M13_M24','M25_M36','M37_PLUS']).astype(str)
    frame['negative_truth']=actual<0
    frame['near_zero_truth']=actual.abs()<=5
    frame['positive_truth']=actual>5
    frame['shape_tail_truth']=(actual-actual_level).abs()>tail_thresholds['shape_abs_p95']
    frame['high_price_truth']=actual>tail_thresholds['price_p95']
    return frame.loc[flags.midnight|flags.month_boundary|flags.season_boundary]


def revision_frame(earlier,later):
    common=earlier.index.intersection(later.index)
    # Saved curves are complete months; no resampling or extrapolation of overlap.
    delta=later.loc[common]-earlier.loc[common]
    if len(delta):
        np.testing.assert_allclose(delta.final,delta.level+delta.assembly_shape+delta.projection,atol=1e-9,rtol=0)
    return delta


def event_summary(events):
    rows=[]
    for (origin,role,candidate),part in events.groupby(['origin','role','candidate'],sort=False):
        kinds={'MONTH_BOUNDARY':part.month_boundary,'SEASON_BOUNDARY':part.season_boundary,
               'OTHER_MONTH_BOUNDARY':part.month_boundary&~part.season_boundary,
               'WITHIN_MONTH_MIDNIGHT':part.midnight&~part.month_boundary}
        segments={'ALL':np.ones(len(part),bool)}
        for name in ['negative_truth','near_zero_truth','positive_truth','shape_tail_truth','high_price_truth']:
            segments[name]=part[name]
        for column in ['delivery_year','season','horizon']:
            for value in sorted(part[column].unique()): segments[f'{column}:{value}']=part[column].eq(value)
        for kind,kindmask in kinds.items():
            for segment,mask in segments.items():
                subset=part.loc[kindmask&mask]
                for field in ERRORS:
                    values=subset.loc[subset.scored,field].dropna().to_numpy()
                    rows.append(dict(origin=origin,role=role,candidate=candidate,kind=kind,segment=segment,error_type=field,
                        events=len(subset),scored_events=len(values),mae=float(np.mean(np.abs(values))) if len(values) else np.nan,
                        rmse=float(np.sqrt(np.mean(values**2))) if len(values) else np.nan,
                        bias=float(np.mean(values)) if len(values) else np.nan,
                        p95=float(np.quantile(np.abs(values),.95)) if len(values) else np.nan,
                        status='SCORED_RETROSPECTIVE' if len(values) else 'UNSUPPORTED'))
    return pd.DataFrame(rows)


def recommendation(events):
    part=events.loc[events.role.eq('assessment')&events.candidate.eq('signed-equal')&events.scored]
    seasonal=part.loc[part.season_boundary]
    other=part.loc[part.month_boundary&~part.season_boundary]
    support=len(seasonal)>=8 and len(other)>=8 and seasonal.origin.nunique()>=3 and other.origin.nunique()>=3
    ratios={field:float(seasonal[field].abs().mean()/other[field].abs().mean())
            if len(other) and other[field].abs().mean()>0 else None for field in ['full_ramp_error','shape_ramp_error']}
    raw=float(seasonal.raw_step.abs().sum())
    total=raw+float(seasonal.centering_step.abs().sum())+float(seasonal.projection_step.abs().sum())
    share=raw/total if total>0 else None
    warranted=bool(support and all(x is not None and x>1.1 for x in ratios.values()) and share is not None and share>.5)
    return dict(smooth_calendar_hypothesis_warranted=warranted,support=bool(support),
        seasonal_events=len(seasonal),other_month_events=len(other),seasonal_origins=int(seasonal.origin.nunique()),
        other_month_origins=int(other.origin.nunique()),mae_ratios=ratios,raw_absolute_contribution_share=share,
        adoption=False,authority=dict(AUTHORITIES))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output.resolve()
    if Path.cwd()!=ROOT or not out.is_relative_to(WORK) or out==WORK:
        raise ValueError('fresh canonical audit output required')
    for name in ('TEMP','TMP','APPDATA','LOCALAPPDATA','MPLCONFIGDIR','XDG_CACHE_HOME','NUMBA_CACHE_DIR','JOBLIB_TEMP_FOLDER','PYTHONUSERBASE','PIP_CACHE_DIR'):
        if not Path(os.environ.get(name,'')).resolve().is_relative_to(WORK): raise ValueError('task-local runtime required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1': raise ValueError('CPU required')
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(filename=out/'runtime.log',level=logging.INFO)
    if sha(PRIOR/'manifest.json')!='1f131effbce0af30e418aa9f47d72bb097c04746e23c7cd68feec9ffeecb0628':
        raise ValueError('D308 manifest mismatch')
    previous=json.loads((PRIOR/'plan.json').read_text())
    pins=dict(previous['inputs_sha256'])
    pins[(PRIOR/'manifest.json').relative_to(ROOT).as_posix()]=sha(PRIOR/'manifest.json')
    for name,digest in json.loads((PRIOR/'manifest.json').read_text()).items(): pins[(PRIOR/name).relative_to(ROOT).as_posix()]=digest
    code=dict(previous['code_sha256'])
    for name,digest in {**pins,**code}.items():
        if sha(ROOT/name)!=digest: raise ValueError(f'prior bytes changed: {name}')
    for name in ('scripts/audit_lt_hourly_revisions.py','scripts/verify_lt_hourly_revisions.py',
                 'tests/test_hourly_revisions.py','docs/model/LT-HOURLY-REVISION-SEAM-AUDIT.md'):
        code[name]=sha(ROOT/name)
    for name in code:
        dest=out/'source-snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,dest)
    origins=previous['origins']
    scales={s['id']:preorigin_scales(pd.read_parquet(PRIOR/s['id']/'targets.parquet'),pd.Timestamp(s['origin_utc'])) for s in origins}
    oldplan=json.loads((OLD/'plan.json').read_text())
    if oldplan['source_policy']!='LATEST_OBSERVED_RETROSPECTIVE_NOT_PIT': raise ValueError('source classification changed')
    plan=dict(schema='fmv-hourly-revision-audit.v1',frozen_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        origins=origins,candidates=previous['controls']+list(previous['candidates']),scales=scales,
        tail_thresholds=previous['thresholds'],pairs=[dict(earlier=a['id'],later=b['id']) for a,b in zip(origins[:-1],origins[1:])],
        source_policy=oldplan['source_policy'],real_daily_D304_vintage_status='UNSUPPORTED_NO_ADMITTED_ARCHIVED_SERIES',
        inputs_sha256=pins,code_sha256=code,authority=dict(AUTHORITIES),max_counterfactual_assemblies=6,
        statistical_estimator_fits=0)
    write_json(out/'plan.json',plan)
    print(json.dumps(dict(stage='plan_frozen',sha256=sha(out/'plan.json'))),flush=True)
    inventory=[]
    all_events=[]
    parts={}
    levels_by_origin={}
    for spec in origins:
        label,origin=spec['id'],pd.Timestamp(spec['origin_utc'])
        solver=json.loads((SOURCE/'monthly-solver/result.json' if label=='current' else OLD/f'{label}/solver.json').read_text())
        levels=solver['assembler_base_prices'] if label=='current' else solver['base_prices']
        levels_by_origin[label]=(solver,levels)
        actual=(pd.Series(dtype=float,index=pd.DatetimeIndex([],tz='UTC')) if label=='current' else
                pd.read_parquet(PRIOR/label/'signed-equal/diagnostics-final.parquet').actual)
        for candidate in plan['candidates']:
            folder=PRIOR/label/candidate
            curve=pd.read_parquet(folder/'curve.parquet')
            p=components(curve,levels)
            parts[(label,candidate)]=p
            raw=pd.read_parquet(folder/'raw.parquet').raw if candidate!='mlp-control' else None
            events=event_frame(p,raw,actual,origin,scales[label],previous['thresholds'][label])
            events['origin'],events['role'],events['candidate']=label,spec['role'],candidate
            all_events.append(events)
            inventory.append(dict(origin=label,candidate=candidate,valuation=origin.isoformat(),
                classification='CURRENT_SINGLE_VALUATION' if label=='current' else 'RETROSPECTIVE_NOT_PIT',
                rows=len(curve),start=curve.index[0].isoformat(),end=curve.index[-1].isoformat(),
                curve_path=(folder/'curve.parquet').relative_to(ROOT).as_posix(),curve_sha256=sha(folder/'curve.parquet')))
        print(json.dumps(dict(stage='seams_saved',origin=label)),flush=True)
    events=pd.concat(all_events)
    events.to_parquet(out/'events.parquet')
    events.reset_index(names='timestamp_utc').to_csv(out/'events.csv',index=False)
    event_summary(events).to_csv(out/'seam-metrics.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False)
    write_json(out/'recommendation.json',recommendation(events))
    revisions=[]
    pair_status=[]
    receipts=[]
    from pfc_shaping.lt.benchmark_safeguards import ForbiddenHourlyModel
    model=ForbiddenHourlyModel()
    (out/'revisions').mkdir()
    (out/'counterfactuals').mkdir()
    for pair in plan['pairs']:
        a,b=pair['earlier'],pair['later']
        name=f'{a}-to-{b}'
        for candidate in plan['candidates']:
            delta=revision_frame(parts[(a,candidate)],parts[(b,candidate)])
            if not len(delta): continue
            delta.to_parquet(out/'revisions'/f'{name}-{candidate}.parquet')
            for year in ['ALL',*sorted(set(delta.index.tz_convert('Europe/Zurich').year))]:
                selected=delta if year=='ALL' else delta.loc[delta.index.tz_convert('Europe/Zurich').year==year]
                for field in delta:
                    scale=scales[a]['revision_p95']
                    revisions.append(dict(pair=name,candidate=candidate,delivery_year=str(year),component=field,hours=len(selected),
                        mean=float(selected[field].mean()),mean_absolute=float(selected[field].abs().mean()),
                        p95_absolute=float(selected[field].abs().quantile(.95)),maximum_absolute=float(selected[field].abs().max()),
                        input_shape_scale=scale,shape_exceedances=int((selected[field].abs()>scale).sum()) if field=='assembly_shape' and scale is not None else None))
        common=parts[(a,'signed-equal')].index.intersection(parts[(b,'signed-equal')].index)
        pair_status.append(dict(pair=name,hours=len(common),status='RETROSPECTIVE_OVERLAP' if len(common) else 'UNSUPPORTED_NO_COMMON_DELIVERY'))
        if not len(common): continue
        newcurve=pd.read_parquet(PRIOR/b/'signed-equal/curve.parquet')
        target=pd.read_parquet(PRIOR/a/'targets.parquet')
        raw,_=calendar_cell_reference(target.signed_target,newcurve.index[::4])
        origin=pd.Timestamp(next(s['origin_utc'] for s in origins if s['id']==b))
        solver,levels=levels_by_origin[b]
        surface=(select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet'))
                 if b=='current' else pd.read_parquet(OLD/f'{b}/eex-surface.parquet'))
        assembly=assembler(model)
        curve=assembly.build(base_prices=levels,quoted_keys=set(solver['quoted_keys']),delivery_index=newcurve.index,
            reference_date=origin,country='CH',signed_hourly_shape=center_signed_hourly_shape(pd.Series(raw,index=newcurve.index[::4])))
        dest=out/'counterfactuals'/name
        receipts.append(dict(pair=name,**save_curve(dest,curve,levels,surface,assembly)))
        pd.Series(raw,index=newcurve.index[::4],name='raw').to_frame().to_parquet(dest/'raw.parquet')
        cf=components(curve,levels)
        before,after=parts[(a,'signed-equal')].loc[common],parts[(b,'signed-equal')].loc[common]
        c=cf.loc[common]
        attribution=pd.DataFrame(dict(total=after.final-before.final,
            market_level=c.level-before.level,grid_shape=c.assembly_shape-before.assembly_shape,
            market_projection=c.projection-before.projection,history_shape=after.assembly_shape-c.assembly_shape,
            history_projection=after.projection-c.projection,history_final=after.final-c.final,market_grid_final=c.final-before.final))
        np.testing.assert_allclose(attribution.total,attribution.market_grid_final+attribution.history_final,atol=1e-9,rtol=0)
        attribution.to_parquet(dest/'attribution.parquet')
        write_json(dest/'counterfactual.json',dict(status='COUNTERFACTUAL_NOT_ARCHIVED_VINTAGE',earlier=a,later=b,
            target_sha256=sha(PRIOR/a/'targets.parquet'),full_grid_hours=len(cf),common_delivery_hours=len(common),authority=dict(AUTHORITIES)))
        if b=='current':
            main=curve.loc[curve.index<pd.Timestamp('2030-01-01',tz='Europe/Zurich')]
            for resolution,series in [('1h',main.price_shape.resample('h').mean()),('15min',main.price_shape)]:
                pd.DataFrame(dict(timestamp_utc=series.index.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    timestamp_ch=series.index.tz_convert('Europe/Zurich').map(lambda t:t.isoformat()),price_eur_mwh=series.to_numpy())).to_csv(
                    dest/f'counterfactual-ch-{resolution}.csv',sep=';',index=False,float_format='%.10f')
        print(json.dumps(dict(stage='counterfactual_complete',pair=name)),flush=True)
    pd.DataFrame(revisions).to_csv(out/'revision-summary.csv',index=False)
    write_json(out/'pair-status.json',pair_status)
    write_json(out/'prospective-holdout-draft.json',dict(status='DRAFT_NOT_INDEPENDENTLY_REGISTERED',
        candidate='D304_signed_equal',forecast_valuation='2026-09-07T08:00:00+00:00',
        forecast_path=(PRIOR/'current/signed-equal/curve.parquet').relative_to(ROOT).as_posix(),
        forecast_sha256=sha(PRIOR/'current/signed-equal/curve.parquet'),
        delivery_start='2026-10-01T00:00:00+02:00',delivery_end_exclusive='2027-10-01T00:00:00+02:00',
        independent_custodian=None,truth_finalization_contract=None,
        blockers=['INDEPENDENT_REGISTRATION','GOVERNED_TRUTH_AVAILABILITY_AND_FINALIZATION'],
        scope='ONE_YEAR_NEAR_HORIZON_NOT_THREE_YEAR_QUALIFICATION',authority=dict(AUTHORITIES)))
    for name,digest in {**pins,**code}.items():
        if sha(ROOT/name)!=digest: raise ValueError(f'bound bytes changed: {name}')
    write_json(out/'complete.json',dict(status='COMPLETE_AUDIT_NO_ADOPTION',counterfactual_assemblies=len(receipts),
        preorigin_calendar_calculations=2*sum(s['revision_p95'] is not None for s in scales.values()),
        counterfactual_calendar_calculations=len(receipts),
        statistical_estimator_fits=0,receipts=receipts,authority=dict(AUTHORITIES),completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    logging.shutdown()
    write_json(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})


if __name__=='__main__': main()
