import numpy as np
import pandas as pd
import pytest

from scripts.audit_lt_hourly_revisions import transitions, preorigin_scales, components, event_frame, revision_frame, recommendation


def history(start='2023-01-01',end='2025-01-01'):
    index=pd.date_range(start,end,freq='h',inclusive='left',tz='Europe/Zurich').tz_convert('UTC')
    price=pd.Series(50+30*np.sin(np.arange(len(index))/4)+np.arange(len(index))/200,index=index)
    months=index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    return pd.DataFrame(dict(price_eur_mwh=price,signed_target=price-price.groupby(months).transform('mean')))


def test_month_season_dst_and_leap_flags():
    index=history('2024-01-01','2025-01-01').index
    flags=transitions(index)
    assert flags.contiguous.sum()==366*24-1
    assert flags.month_boundary.sum()==11 and flags.season_boundary.sum()==4
    assert flags.midnight.sum()==365
    # FMV operational seasons, not meteorological quarters: Apr/Jun/Oct/Nov.
    assert set(index[flags.season_boundary].tz_convert('Europe/Zurich').month)=={4,6,10,11}


def test_gap_is_not_a_midnight_or_season_event():
    index=pd.DatetimeIndex(['2024-02-29T21:00Z','2024-02-29T23:00Z','2024-03-01T00:00Z'])
    flags=transitions(index)
    assert not flags.iloc[1].any()
    assert flags.contiguous.iloc[2]


@pytest.mark.parametrize('kind',['naive','duplicate','reversed'])
def test_bad_transition_index_rejected(kind):
    index=history().index
    if kind=='naive': index=index.tz_localize(None)
    elif kind=='duplicate': index=index.append(index[-1:])
    else: index=index[::-1]
    with pytest.raises(ValueError): transitions(index)


def test_preorigin_budget_and_short_history_support():
    target=history()
    origin=pd.Timestamp('2025-01-01T00:00Z')
    scales=preorigin_scales(target,origin)
    assert scales['boundary_count']==23 and scales['revision_probe_hours']==365*24
    assert scales['revision_p95']>=0 and pd.Timestamp(scales['older_training_end'])<origin
    short=preorigin_scales(history('2024-07-01','2025-01-01'),origin)
    assert short['boundary_p95'] is None and short['revision_status']=='UNSUPPORTED'


@pytest.mark.parametrize('kind',['future','gap','wrong_signed'])
def test_threshold_history_contract(kind):
    target=history()
    origin=pd.Timestamp('2025-01-01T00:00Z')
    if kind=='future': origin=target.index[-1]
    elif kind=='gap': target=target.drop(target.index[30])
    else: target.signed_target+=10
    with pytest.raises(ValueError): preorigin_scales(target,origin)


def test_boundary_decomposition_does_not_remove_level_jump():
    index=pd.date_range('2024-01-31T21:00Z',periods=4,freq='h')
    parts=pd.DataFrame(dict(final=[40.,45.,100.,105.],level=[50.,50.,100.,100.],
                            assembly_shape=[-12.,-7.,-2.,3.],projection=[2.]*4),index=index)
    actual=pd.Series([35.,40.,48.,55.],index=index)
    events=event_frame(parts,pd.Series([-10.,-5.,0.,5.],index=index),actual,pd.Timestamp('2024-01-01T00:00Z'),
        dict(boundary_p95=20.,midnight_p95=10.),dict(shape_abs_p95=10.,price_p95=50.))
    assert len(events)==1 and events.month_boundary.iloc[0]
    assert events.final_step.iloc[0]==55 and events.level_step.iloc[0]==50
    assert events.forecast_step_exceeds_scale.iloc[0]
    np.testing.assert_allclose(events.full_ramp_error,events.shape_ramp_error+events.level_ramp_error,atol=1e-12)
    np.testing.assert_allclose(events.final_step,events.level_step+events.raw_step+events.centering_step+events.projection_step,atol=1e-12)


def test_components_solver_missing_fails_and_projection_is_separate():
    index=pd.date_range('2024-01-01T00:00Z',periods=8,freq='15min')
    curve=pd.DataFrame(dict(price_shape=[45.]*4+[65.]*4,price_pre_final_projection=[40.]*4+[60.]*4),index=index)
    result=components(curve,{'2024-01':50.})
    np.testing.assert_array_equal(result.projection,[5.,5.])
    with pytest.raises(ValueError): components(curve,{})


def test_revision_intersection_and_exact_components():
    index=pd.date_range('2025-01-01T00:00Z',periods=48,freq='h')
    a=pd.DataFrame(dict(final=60.,level=50.,assembly_shape=8.,projection=2.),index=index)
    b=pd.DataFrame(dict(final=83.,level=70.,assembly_shape=10.,projection=3.),index=index[12:])
    delta=revision_frame(a,b)
    assert len(delta)==36
    np.testing.assert_array_equal(delta.final,[23.]*36)
    np.testing.assert_allclose(delta.final,delta.level+delta.assembly_shape+delta.projection)
    b.index+=pd.Timedelta(days=100)
    assert revision_frame(a,b).empty


def test_future_events_remain_unscored():
    index=pd.date_range('2027-03-01',periods=48,freq='h',tz='Europe/Zurich').tz_convert('UTC')
    parts=pd.DataFrame(dict(final=50.,level=50.,assembly_shape=0.,projection=0.),index=index)
    events=event_frame(parts,None,pd.Series(dtype=float,index=pd.DatetimeIndex([],tz='UTC')),
        pd.Timestamp('2026-09-07T08:00Z'),dict(boundary_p95=None,midnight_p95=10.),dict(shape_abs_p95=20.,price_p95=100.))
    assert not events.scored.any() and events.full_ramp_error.isna().all()


@pytest.mark.parametrize('veto',['none','support','full_only','projection'])
def test_single_hypothesis_requires_all_frozen_conditions(veto):
    rows=[]
    for origin in ['2023','2024','2025','2026']:
        for season in [True,False]:
            for _ in range(2):
                rows.append(dict(role='assessment',origin=origin,candidate='signed-equal',scored=True,
                    season_boundary=season,month_boundary=True,full_ramp_error=12. if season else 10.,
                    shape_ramp_error=12. if season else 10.,raw_step=10.,centering_step=1.,projection_step=1.))
    frame=pd.DataFrame(rows)
    if veto=='support': frame=frame.loc[frame.origin.eq('2023')]
    elif veto=='full_only': frame.shape_ramp_error=10.
    elif veto=='projection': frame.projection_step=100.
    result=recommendation(frame)
    assert result['smooth_calendar_hypothesis_warranted']==(veto=='none')
    assert result['adoption'] is False and not any(result['authority'].values())
