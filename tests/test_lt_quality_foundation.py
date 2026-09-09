import numpy as np
import pandas as pd
import pytest

from pfc_shaping.validation.lt_source_quality import delivery_hours, reconstruct_quote_conflicts, assess_eex_freshness
from pfc_shaping.validation.lt_economic_profiles import value_fixed_profile


def quote_fixture():
    surface = pd.DataFrame([dict(product=p, load_type='BASE', price=v, date='2026-09-07',
        market='CH',unit='EUR/MWH',fact_load_timestamp_utc='2026-09-08T02:00:00Z')
        for p,v in [('2026-10',10.),('2026-11',20.),('2026-12',30.),('2026-Q4',20.)]])
    weighted = sum(v*len(delivery_hours(p,'BASE')) for p,v in [('2026-10',10.),('2026-11',20.),('2026-12',30.)])/2209
    gates = pd.DataFrame([dict(product=p,load_type='BASE',gate_id='hard_base_product_repricing',status='PASS',
        residual_eur_mwh=0.,covered_by_quote_aware_products='') for p in ['2026-10','2026-11','2026-12']]+[
        dict(product='2026-Q4',load_type='BASE',gate_id='hard_base_product_repricing',status='QUOTE_CONFLICT',
             residual_eur_mwh=weighted-20.,covered_by_quote_aware_products='2026-10,2026-11,2026-12')])
    return gates,surface


def test_swiss_dst_and_eex_peak_hours():
    assert len(delivery_hours('2026-10','BASE')) == 745
    assert len(delivery_hours('2027-03','BASE')) == 743
    assert len(delivery_hours('2026-Q4','PEAK')) == 792
    assert len(delivery_hours('2026-Q4','OFFPEAK')) == 1417


def test_independent_hour_weights_and_no_waiver():
    gates,surface = quote_fixture()
    result = reconstruct_quote_conflicts(gates,surface)
    assert result['rows'][0]['reconstructed_mean'] == pytest.approx((745*10+720*20+744*30)/2209)
    assert result['accepted_conflicts'] == 0
    assert not any(result['authority'].values())


@pytest.mark.parametrize('children',['2026-10,2026-11','2026-10,2026-10,2026-12','2026-Q4'])
def test_reject_incomplete_overlapping_or_parent_children(children):
    gates,surface = quote_fixture()
    gates.loc[3,'covered_by_quote_aware_products'] = children
    with pytest.raises(ValueError): reconstruct_quote_conflicts(gates,surface)


def test_do_not_explain_critical_child_as_rounding():
    gates,surface = quote_fixture()
    gates.loc[0,'status']='CRITICAL'
    with pytest.raises(ValueError,match='must pass'): reconstruct_quote_conflicts(gates,surface)


def test_do_not_hide_unexplained_curve_error():
    gates,surface = quote_fixture()
    gates.loc[3,'residual_eur_mwh'] += 1.
    with pytest.raises(ValueError,match='reproduce'): reconstruct_quote_conflicts(gates,surface)


def test_freshness_keeps_publication_and_observation_separate():
    _,surface = quote_fixture()
    result = assess_eex_freshness(surface,observed_at='2026-09-08T10:00Z',valuation_at='2026-09-08T11:00Z',expected_quote_date='2026-09-07')
    assert result.status.eq('EXPECTED_DATE_OBSERVED').all()
    assert not result.independent_availability_proven.any()
    surface.loc[0,'date']='2026-09-04'
    surface.loc[1,'fact_load_timestamp_utc']='2026-09-08T12:00Z'
    result = assess_eex_freshness(surface,observed_at='2026-09-08T10:00Z',valuation_at='2026-09-08T11:00Z',expected_quote_date='2026-09-07')
    assert result.status.tolist()[:2] == ['STALE_QUOTE','POST_OBSERVATION_LOAD']


def profile_fixture(minutes=60):
    idx = pd.date_range('2026-10-01',periods=4,freq=f'{minutes}min',tz='UTC')
    prices = pd.Series([-10.,0.,10.,20.],index=idx)
    profile = pd.DataFrame(dict(timestamp_utc=idx,volume_mwh=[1.,2.,3.,4.],direction='GENERATION',
        profile_id='TEST_ONLY',profile_version='v1',available_at_utc='2026-09-08T09:00Z',source_document_id='synthetic-test'))
    return prices,profile


@pytest.mark.parametrize('minutes',[15,60])
def test_mwh_cashflow_sign_and_negative_prices(minutes):
    prices,profile = profile_fixture(minutes)
    result = value_fixed_profile(prices,profile,valuation_at='2026-09-08T10:00Z',native_interval_minutes=minutes)
    assert result['signed_cashflow_eur'] == 100
    assert result['capture_price_eur_mwh'] == 10
    assert result['negative_price_mwh'] == 1
    profile.direction='CONSUMPTION'
    assert value_fixed_profile(prices,profile,valuation_at='2026-09-08T10:00Z',native_interval_minutes=minutes)['signed_cashflow_eur'] == -100


@pytest.mark.parametrize('mutation',['late','negative_volume','missing','mixed','nan'])
def test_reject_bad_economic_inputs(mutation):
    prices,profile = profile_fixture()
    if mutation=='late': profile.loc[0,'available_at_utc']='2026-09-09T00:00Z'
    if mutation=='negative_volume': profile.loc[0,'volume_mwh']=-1
    if mutation=='missing': profile=profile.drop(1)
    if mutation=='mixed': profile.loc[0,'profile_version']='v2'
    if mutation=='nan': prices.iloc[0]=np.nan
    with pytest.raises(ValueError): value_fixed_profile(prices,profile,valuation_at='2026-09-08T10:00Z',native_interval_minutes=60)


def test_zero_volume_is_unsupported_not_pass():
    prices,profile = profile_fixture()
    profile.volume_mwh=0.
    result=value_fixed_profile(prices,profile,valuation_at='2026-09-08T10:00Z',native_interval_minutes=60)
    assert result['status']=='UNSUPPORTED_ZERO_VOLUME'
    assert result['capture_price_eur_mwh'] is None


def test_no_blk13_or_assumed_population_requirement():
    prices,profile = profile_fixture()
    profile.profile_id='CONFIRMED_FMV_PROFILE'
    result=value_fixed_profile(prices,profile,valuation_at='2026-09-08T10:00Z',native_interval_minutes=60)
    assert result['status']=='LOCAL_VALUATION_NOT_ECONOMIC_ADMISSION'
    assert all(value is False for value in result['authority'].values())
