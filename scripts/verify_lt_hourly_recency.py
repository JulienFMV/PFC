"""Independent D307 saved-artifact verification, comparison and report."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.databricks_eex_daily_snapshot import select_latest_quote_surface
from pfc_shaping.validation.product_normalization import build_product_normalization_gates
from scripts.run_lt_signed_composition import ROOT, SOURCE, OLD, sha, write_json

D305=ROOT/'build/lt-signed-composition-20260907/run-v4'


def independent_prediction(target,weights,delivery):
    train=enrich_15min_index(target.index)
    train['value'],train['weight']=target.to_numpy(),weights.to_numpy()
    future=enrich_15min_index(delivery)
    result=np.full(len(delivery),np.nan)
    counts={}
    for keys in [('saison','type_jour','heure_hce'),('saison','heure_hce'),('heure_hce',)]:
        table={}
        for key,part in train.groupby(list(keys)):
            key=key if isinstance(key,tuple) else (key,)
            table[key]=float(np.average(part.value,weights=part.weight))
        values=np.array([table.get(tuple(row),np.nan) for row in future[list(keys)].itertuples(index=False,name=None)])
        use=np.isnan(result)&np.isfinite(values)
        result[use]=values[use]
        counts['/'.join(keys)]=int(use.sum())
    counts['global']=int(np.isnan(result).sum())
    result[np.isnan(result)]=float(np.average(target,weights=weights))
    return result,counts


def independent_masks(index,truth,ramp,origin,thresholds):
    local=index.tz_convert('Europe/Zurich')
    cal=enrich_15min_index(index)
    lead=(local.year-origin.tz_convert('Europe/Zurich').year)*12+local.month-origin.tz_convert('Europe/Zurich').month
    peak=(local.dayofweek<5)&(local.hour>=8)&(local.hour<20)
    result={'ALL':np.ones(len(index),bool),'NEGATIVE_TRUTH':truth<0,'NEAR_ZERO_TRUTH':np.abs(truth)<=5,
        'POSITIVE_TRUTH':truth>5,'HIGH_PRICE':truth>thresholds['price_p95'],
        'LARGE_RAMP':np.abs(ramp)>thresholds['ramp_p95'],'PEAK_CLOCK':peak,'OFFPEAK_CLOCK':~peak,
        'WEEKEND_OR_HOLIDAY':cal.type_jour.ne('Ouvrable').to_numpy(),'COMMON_FIRST8':(lead>=1)&(lead<=8)}
    for a,b in [(1,6),(7,12),(13,24),(25,36)]: result[f'M{a:02d}_M{b:02d}']=(lead>=a)&(lead<=b)
    for season in ['Hiver','Printemps','Ete','Automne']: result[season]=cal.saison.eq(season).to_numpy()
    for year in sorted(set(local.year)): result[f'YEAR_{year}']=local.year==year
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run,out=args.run.resolve(),args.output.resolve()
    task=ROOT/'build/lt-hourly-recency-20260908'
    if Path.cwd()!=ROOT or not run.is_relative_to(task) or not out.is_relative_to(task) or out==task:
        raise ValueError('canonical task paths required')
    out.mkdir(exist_ok=False)
    plan=json.loads((run/'plan.json').read_text())
    complete=json.loads((run/'complete.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    for name,digest in manifest.items(): assert sha(run/name)==digest,name
    for name,digest in {**plan['inputs_sha256'],**plan['code_sha256']}.items(): assert sha(ROOT/name)==digest,name
    assert not any(complete['authority'].values()) and not any(plan['authority'].values())
    truth_qh=pd.read_parquet(SOURCE/'prepared-inputs/epex-ch.parquet').price_eur_mwh
    assert (truth_qh.resample('h').max()-truth_qh.resample('h').min()).eq(0).all()
    truth=truth_qh.resample('h').mean()
    saved_metrics=pd.read_csv(run/'diagnostic-metrics.csv',dtype={'origin':str}).set_index(['origin','candidate','stage','segment','error_type'])
    old_metrics=pd.read_csv(run/'native-metrics.csv',dtype={'origin':str}).set_index(['origin','candidate','stage','segment'])
    month_metrics=pd.read_csv(run/'monthly-decomposition.csv',dtype={'origin':str}).set_index(['origin','candidate','stage','month'])
    verified,monthly,profiles,replays=[],[],[],[]
    max_monthly=max_diagnostic=0.
    origins=plan['folds']+[dict(id='current',role='descriptive',origin_utc='2026-09-07T08:00:00+00:00')]
    for spec in origins:
        label,origin=spec['id'],pd.Timestamp(spec['origin_utc'])
        target=pd.read_parquet(run/label/'targets.parquet')
        assert target.index.max()<origin
        np.testing.assert_array_equal(target.price_eur_mwh,truth.reindex(target.index))
        groups=target.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
        np.testing.assert_allclose(target.signed_target,target.price_eur_mwh-target.price_eur_mwh.groupby(groups).transform('mean'),atol=1e-12,rtol=0)
        for month,part in target.groupby(groups):
            start=pd.Timestamp(month+'-01',tz='Europe/Zurich')
            end=(pd.Period(month,freq='M')+1).start_time.tz_localize('Europe/Zurich')
            assert end<=origin and part.index.equals(pd.date_range(start,end,freq='h',inclusive='left').tz_convert('UTC'))
        thresholds=json.loads((run/label/'thresholds.json').read_text())
        train_ramps=target.price_eur_mwh.diff().to_numpy()
        train_ramps[np.r_[True,groups[1:]!=groups[:-1]]]=np.nan
        np.testing.assert_allclose([thresholds['price_p95'],thresholds['ramp_p95']],
            [np.quantile(target.price_eur_mwh,.95),np.nanquantile(np.abs(train_ramps),.95)],atol=1e-12,rtol=0)
        solver=json.loads((SOURCE/'monthly-solver/result.json' if label=='current' else OLD/f'{label}/solver.json').read_text())
        levels=solver['assembler_base_prices'] if label=='current' else solver['base_prices']
        surface=(select_latest_quote_surface(pd.read_parquet(SOURCE/'eex-replay/eex-normalized-history.parquet')) if label=='current' else pd.read_parquet(OLD/f'{label}/eex-surface.parquet'))
        population=None
        baseline=None
        for candidate in ['mlp-control']+list(plan['candidates']):
            folder=run/label/candidate
            curve=pd.read_parquet(folder/'curve.parquet')
            index=curve.index
            assert index.is_unique and np.isfinite(curve.price_shape).all()
            if candidate=='mlp-control':
                prior=pd.read_parquet(D305/('current/mlp-current/curve.parquet' if label=='current' else f'{label}/mlp/curve.parquet'))
                pd.testing.assert_frame_equal(curve,prior)
            else:
                weights=pd.read_parquet(folder/'weights.parquet').weight
                assert weights.index.equals(target.index)
                half=plan['candidates'][candidate]
                expected=np.ones(len(target)) if half is None else np.exp2(-(target.index[-1]-target.index).total_seconds().to_numpy()/86400/half)
                np.testing.assert_array_equal(weights,expected)
                raw,counts=independent_prediction(target.signed_target,weights,index[::4])
                saved_raw=pd.read_parquet(folder/'raw.parquet').raw
                error=float(np.max(np.abs(raw-saved_raw.to_numpy())))
                assert error<1e-10
                replays.append(error)
                fit=json.loads((folder/'fit.json').read_text())
                assert counts==fit['counts'] and fit['rows']==len(target)
                assert abs(fit['global_effective_hours']-weights.sum()**2/(weights**2).sum())<1e-8
                # Saved signed input must be the independently reconstructed raw
                # shape centered over the full predicted Swiss month.
                raw_series=pd.Series(raw,index=index[::4])
                months=raw_series.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
                centered=raw_series-raw_series.groupby(months).transform('mean')
                np.testing.assert_allclose(curve.signed_hourly_shape_eur_mwh.iloc[::4],centered,atol=1e-10,rtol=0)
                np.testing.assert_allclose(np.ptp(curve.price_shape.to_numpy().reshape(-1,4),axis=1),0,atol=1e-10,rtol=0)
                if candidate=='signed-equal':
                    baseline=curve
                    prior=pd.read_parquet(D305/('current/signed/curve.parquet' if label=='current' else f'{label}/signed/curve.parquet'))
                    np.testing.assert_allclose(curve.price_shape,prior.price_shape,atol=1e-9,rtol=0)
                monthly_means=curve.price_shape.groupby(index.tz_convert('Europe/Zurich').strftime('%Y-%m')).mean()
                drift=float((monthly_means-pd.Series(levels).reindex(monthly_means.index)).abs().max())
                assert drift<1e-9
                max_monthly=max(max_monthly,drift)
                h=curve.price_shape.resample('h').mean().to_frame('price_eur_mwh')
                h['ts_ch']=h.index.tz_convert('Europe/Zurich')
                for field in ['year','month','quarter']: h[field]=getattr(h.ts_ch.dt,field)
                gates=build_product_normalization_gates(h,surface,forward_date=pd.Timestamp(surface.date.iloc[0]),price_column='price_eur_mwh',hard_tolerance=1e-6,peak_country='CH')
                saved=pd.read_csv(folder/'product-gates.csv')
                for column in gates.select_dtypes(include='object'):
                    gates[column]=gates[column].fillna(''); saved[column]=saved[column].fillna('')
                pd.testing.assert_frame_equal(gates.reset_index(drop=True),saved,check_dtype=False,atol=1e-9,rtol=1e-10)
                assert not gates.status.eq('CRITICAL').any()
                if label=='current': assert gates.status.value_counts().to_dict()=={'PASS':80,'QUOTE_CONFLICT':9}
            if label=='current':
                main=curve.loc[index<pd.Timestamp('2030-01-01',tz='Europe/Zurich')]
                for resolution,series in [('1h',main.price_shape.resample('h').mean()),('15min',main.price_shape)]:
                    export=pd.read_csv(folder/f'pfc-fmv-ch-{resolution}.csv',sep=';')
                    assert pd.DatetimeIndex(pd.to_datetime(export.timestamp_utc,utc=True)).equals(series.index)
                    assert export.timestamp_ch.tolist()==series.index.tz_convert('Europe/Zurich').map(lambda t:t.isoformat()).tolist()
                    np.testing.assert_allclose(export.price_eur_mwh,series,atol=5.1e-11,rtol=0)
                for year,part in curve.groupby(index.tz_convert('Europe/Zurich').year):
                    p=part.price_shape.resample('h').mean()
                    profiles.append(dict(candidate=candidate,year=int(year),hours=len(p),minimum=float(p.min()),maximum=float(p.max()),negative_hours=int(p.lt(0).sum())))
                continue
            for stage,column in [('final','price_shape'),('pre_projection','price_pre_final_projection')]:
                saved=pd.read_parquet(folder/f'diagnostics-{stage}.parquet')
                idx=saved.index
                if population is None: population=idx
                assert idx.equals(population)
                # Verify every eligible complete month, not merely saved subsets.
                predicted=curve[column].resample('h').mean()
                expected=[]
                for month in sorted(set(predicted.index.tz_convert('Europe/Zurich').strftime('%Y-%m'))):
                    start=pd.Timestamp(month+'-01',tz='Europe/Zurich')
                    end=(pd.Period(month,freq='M')+1).start_time.tz_localize('Europe/Zurich')
                    grid=pd.date_range(start,end,freq='h',inclusive='left').tz_convert('UTC')
                    if end<=pd.Timestamp('2026-09-01',tz='Europe/Zurich') and predicted.reindex(grid).notna().all() and truth.reindex(grid).notna().all(): expected.extend(grid)
                assert idx.equals(pd.DatetimeIndex(expected))
                actual=truth.reindex(idx).to_numpy()
                forecast=predicted.reindex(idx).to_numpy()
                months=idx.tz_convert('Europe/Zurich').strftime('%Y-%m')
                full=forecast-actual
                shape=np.empty(len(idx)); level=np.empty(len(idx))
                for month in sorted(set(months)):
                    where=months==month
                    bias=forecast[where].mean()-actual[where].mean()
                    level[where]=bias; shape[where]=full[where]-bias
                    metrics=dict(hours=int(where.sum()),predicted_mean=float(forecast[where].mean()),actual_mean=float(actual[where].mean()),
                        shape_mse=float(np.mean(shape[where]**2)),level_mse=float(bias**2),full_mse=float(np.mean(full[where]**2)))
                    old=month_metrics.loc[(label,candidate,stage,month)]
                    np.testing.assert_allclose(old[list(metrics)].to_numpy(dtype=float),list(metrics.values()),atol=1e-8,rtol=1e-12)
                    assert abs(metrics['full_mse']-metrics['shape_mse']-metrics['level_mse'])<1e-8
                    monthly.append(dict(origin=label,candidate=candidate,stage=stage,month=month,**metrics))
                ramps=np.r_[np.nan,np.diff(actual)]
                ramp_error=np.r_[np.nan,np.diff(forecast)-np.diff(actual)]
                boundary=np.r_[True,(months[1:]!=months[:-1])|(np.diff(idx.as_unit('ns').asi8)!=3_600_000_000_000)]
                ramps[boundary]=np.nan; ramp_error[boundary]=np.nan
                arrays=dict(actual=actual,prediction=forecast,full_error=full,shape_error=shape,level_error=level,truth_ramp=ramps,ramp_error=ramp_error)
                for field,array in arrays.items():
                    np.testing.assert_allclose(saved[field],array,atol=1e-9,rtol=0,equal_nan=True)
                    max_diagnostic=max(max_diagnostic,float(np.nanmax(np.abs(saved[field].to_numpy()-array))))
                np.testing.assert_allclose(pd.read_parquet(folder/f'errors-{stage}.parquet').iloc[:,0],shape,atol=1e-9,rtol=0)
                if baseline is not None and candidate.startswith('signed'):
                    reference=baseline[column].resample('h').mean().reindex(idx)
                    np.testing.assert_allclose(pd.Series(forecast,index=idx).groupby(months).mean(),reference.groupby(months).mean(),atol=1e-9,rtol=0)
                for segment,mask in independent_masks(idx,actual,ramps,origin,thresholds).items():
                    for field in ['shape_error','full_error','level_error','ramp_error']:
                        selected=arrays[field][mask]
                        selected=selected[np.isfinite(selected)]
                        n=len(selected)
                        row=saved_metrics.loc[(label,candidate,stage,segment,field)]
                        assert int(row.hours)==n
                        values=[float(np.mean(np.abs(selected))),float(np.sqrt(np.mean(selected**2))),float(np.mean(selected)),float(np.quantile(np.abs(selected),.95))] if n else [np.nan]*4
                        np.testing.assert_allclose(row[['mae','rmse','bias','p95']].to_numpy(dtype=float),values,atol=1e-9,rtol=0,equal_nan=True)
                        if field=='shape_error' and (label,candidate,stage,segment) in old_metrics.index:
                            native=old_metrics.loc[(label,candidate,stage,segment)]
                            np.testing.assert_allclose(native[['mae','rmse','bias']].to_numpy(dtype=float),values[:3],atol=1e-9,rtol=0,equal_nan=True)
                        verified.append(dict(origin=label,role=spec['role'],candidate=candidate,stage=stage,segment=segment,error_type=field,
                            hours=n,mae=values[0],rmse=values[1],bias=values[2],p95=values[3],abs_sum=float(np.abs(selected).sum()),sq_sum=float(np.square(selected).sum())))
        print(json.dumps(dict(stage='verified_origin',origin=label)),flush=True)
    metrics=pd.DataFrame(verified)
    metrics.to_csv(out/'verified-metrics.csv',index=False)
    pd.DataFrame(monthly).to_csv(out/'verified-monthly-decomposition.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'current-profiles-by-year.csv',index=False)
    assessment=metrics.loc[metrics.role.eq('assessment')&metrics.stage.eq('final')]
    rows=[]
    for keys,part in assessment.groupby(['candidate','segment','error_type']):
        n=int(part.hours.sum())
        rows.append(dict(candidate=keys[0],segment=keys[1],error_type=keys[2],hours=n,origins=int(part.hours.gt(0).sum()),
            mae=part.mae.mean(),rmse=part.rmse.mean(),pooled_mae=part.abs_sum.sum()/n if n else np.nan,pooled_rmse=np.sqrt(part.sq_sum.sum()/n) if n else np.nan))
    comparison=pd.DataFrame(rows)
    comparison.to_csv(out/'comparison.csv',index=False)
    gates=[]
    for row in comparison.loc[comparison.candidate.isin(['signed-hl365','signed-hl730']) & comparison.error_type.isin(['shape_error','ramp_error'])].itertuples():
        ref=comparison.loc[comparison.candidate.eq('signed-equal') & comparison.segment.eq(row.segment)&comparison.error_type.eq(row.error_type)].iloc[0]
        supported=row.hours>=168 and row.origins>=2
        a=row.mae/ref.mae if ref.mae>0 else np.nan
        b=row.rmse/ref.rmse if ref.rmse>0 else np.nan
        gates.append(dict(candidate=row.candidate,segment=row.segment,error_type=row.error_type,hours=row.hours,origins=row.origins,
            mae_ratio=a,rmse_ratio=b,status='UNSUPPORTED' if not supported else 'REGRESSION' if max(a,b)>1.05 else 'PASS_LOCAL_SCREEN'))
    gateframe=pd.DataFrame(gates)
    gateframe.to_csv(out/'regime-gates.csv',index=False)
    selection=[]
    allshape=assessment.loc[assessment.segment.eq('ALL')&assessment.error_type.eq('shape_error')].pivot(index='origin',columns='candidate',values='mae')
    allshape.to_csv(out/'shape-mae-by-origin.csv')
    aggregate=comparison.loc[comparison.segment.eq('ALL')&comparison.error_type.eq('shape_error')].set_index('candidate')
    for candidate in ['signed-hl365','signed-hl730']:
        wins=int((allshape[candidate]<allshape['signed-equal']).sum())
        a=1-aggregate.loc[candidate,'mae']/aggregate.loc['signed-equal','mae']
        b=1-aggregate.loc[candidate,'rmse']/aggregate.loc['signed-equal','rmse']
        regressions=int((gateframe.candidate.eq(candidate)&gateframe.status.eq('REGRESSION')).sum())
        selection.append(dict(candidate=candidate,mae_gain_vs_signed_pct=100*a,rmse_gain_vs_signed_pct=100*b,wins=wins,
            regime_regressions=regressions,local_screen_pass=bool(a>=.02 and b>=.02 and wins>=3 and regressions==0),adoption=False))
    write_json(out/'screening.json',selection)
    verification=dict(status='VERIFIED',inputs=len(plan['inputs_sha256']),sources=len(plan['code_sha256']),outputs=len(manifest),
        metric_rows=len(metrics),monthly_rows=len(monthly),weighted_reference_replays=len(replays),max_raw_replay_error=max(replays),
        max_diagnostic_error=max_diagnostic,max_monthly_solver_error=max_monthly,authority=plan['authority'],verifier_sha256=sha(Path(__file__)))
    write_json(out/'verification.json',verification)
    report=['# D307 — benchmark horaire CH et pondération historique','',
        'Exécution CPU locale, vérification indépendante terminée. Aucun modèle adopté automatiquement. Références D304/D305/D306 préservées.',
        '', '## Résolution suisse', '',
        'EPEX définit le day-ahead CH en60min dans sa documentation de juillet2026 (p40). JAO a reporté au calendrier2027 le projet15min MTU des enchères de capacité aux frontières ; fin2027 et la date de lancement de l’enchère d’énergie CH restent non confirmées.',
        '[EPEX](https://www.epexspot.com/sites/default/files/download_center_files/EPEX%20SPOT%20Indices%202019-05_final.pdf) · [JAO30juin2026](https://www.jao.eu/news/update-st-auctions-swiss-borders-15-min-mtu-and)',
        '', '## Comparaison sur quatre origines exposées2023–2026', '',
        'Moyennes à poids égal par origine ; les horizons et années se recouvrent. Pas de nouvelle preuve indépendante.', '',
        '| Candidat | Type d’erreur | MAE EUR/MWh | RMSE EUR/MWh |', '|---|---|---:|---:|']
    for row in comparison.loc[comparison.segment.eq('ALL')].itertuples(): report.append(f'| {row.candidate} | {row.error_type} | {row.mae:.6f} | {row.rmse:.6f} |')
    report+=['','## Filtre fixé avant les résultats','',
        'Gain>=2% sur MAE et RMSE de forme versus D304,>=3/4 origines gagnées, aucune régression>5% sur un segment supporté de forme ou rampe. Pas de promotion, même si ce filtre local passe.','']
    for row in selection: report.append(f"- {row['candidate']} : gain MAE{row['mae_gain_vs_signed_pct']:.3f}%, RMSE{row['rmse_gain_vs_signed_pct']:.3f}%, {row['wins']}/4 origines ; {row['regime_regressions']} régressions ; filtre local={row['local_screen_pass']}.")
    report+=['','Les régressions par prix négatifs/proches de zéro, pointes, grandes rampes, saisons, années et horizons sont intégralement conservées dans regime-gates.csv et verified-metrics.csv. comparison.csv conserve les moyennes par origine et les scores pondérés par heure ; COMMON_FIRST8 compare un horizon commun.','',
        '## Niveau, forme et rampes','',
        'Le solveur impose les mêmes niveaux mensuels à tous les candidats. Le recentrage peut améliorer la forme mais ne corrige pas une erreur de niveau mensuel. L’identité MSE totale = MSE forme + MSE niveau est vérifiée pour chaque mois complet. Les rampes excluent les frontières de mois et les trous temporels. Les seuils de pointes/rampes proviennent exclusivement du passé de chaque origine.',
        'Les observations CH sont horaires répétées. Les résultats portent sur une forme horaire, pas sur une précision suisse native à15min, un résultat économique FMV ou une qualification physique2030. Les années exposées, les origines dépendantes et les historiques révisés interdisent une conclusion de promotion scientifique.',
        '', '## Exports', '',
        f'Sous ../{run.name}/current/{{signed-equal,signed-hl365,signed-hl730}}/ : pfc-fmv-ch-1h.csv, pfc-fmv-ch-15min.csv et curve.parquet. Les trois candidats signés répètent chaque prix horaire sur quatre quarts ; le MLP témoin conserve son profil antérieur. Les CSV couvrent Oct2026–Dec2029 ; Parquet conserve le plein horizon jusqu’à2032.',
        'Valorisation retenue du7septembre2026 à08Z, cotations du4septembre : pas de revalorisation au8septembre. Les trois candidats gardent80 PASS/9 QUOTE_CONFLICT, aucun CRITICAL. Tous les champs d’autorité restent false.',
        '', '## Vérification', '',json.dumps(verification,indent=2),'']
    (out/'RAPPORT-COMPARATIF.md').write_text('\n'.join(report),encoding='utf-8')
    write_json(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(verification=verification,screening=selection)),flush=True)


if __name__=='__main__':
    main()
