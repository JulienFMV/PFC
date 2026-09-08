import numpy as np
import pandas as pd
import pytest

from scripts.run_lt_hourly_stability import blend_raw, training_thresholds, stability_masks
from scripts.run_lt_hourly_recency import diagnostic_frame
from scripts.verify_lt_hourly_stability import screen


def history():
    index = pd.date_range('2024-02-01', '2024-04-01', freq='h', inclusive='left', tz='Europe/Zurich').tz_convert('UTC')
    price = pd.Series(20+60*np.cos(np.arange(len(index))/4), index=index)
    months = index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    return pd.DataFrame(dict(price_eur_mwh=price, signed_target=price-price.groupby(months).transform('mean')))


def test_signed_blend_preserves_endpoints_and_monthly_centering():
    target = history().signed_target
    recent = target*2
    pd.testing.assert_series_equal(blend_raw(target, recent, 0), target)
    pd.testing.assert_series_equal(blend_raw(target, recent, 1), recent)
    actual = blend_raw(target, recent, .25)
    np.testing.assert_allclose(actual, target*1.25, atol=1e-12)
    assert actual.min()<0 and len(actual)==60*24-1
    months = target.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    np.testing.assert_allclose(actual.groupby(months).mean(), 0, atol=1e-12)


@pytest.mark.parametrize('bad', ['shift', 'reverse', 'nan', 'inf', 'duplicate', 'negative_alpha', 'large_alpha', 'array_alpha', 'bool_alpha'])
def test_blend_rejects_unpaired_or_invalid_inputs(bad):
    a = history().signed_target
    b, alpha = a.copy(), .25
    if bad=='shift': b.index += pd.Timedelta(hours=1)
    elif bad=='reverse': b = b.iloc[::-1]
    elif bad=='nan': b.iloc[0] = np.nan
    elif bad=='inf': b.iloc[0] = np.inf
    elif bad=='duplicate': a = pd.concat([a, a.iloc[:1]]); b = a.copy()
    elif bad=='negative_alpha': alpha = -.1
    elif bad=='large_alpha': alpha = 1.1
    elif bad=='array_alpha': alpha = np.ones(len(a))*.25
    elif bad=='bool_alpha': alpha = True
    with pytest.raises(ValueError): blend_raw(a, b, alpha)


def test_preorigin_thresholds_signed_tails_and_level_invariance():
    target = history()
    origin = pd.Timestamp('2024-04-01T00:00Z')
    thresholds = training_thresholds(target, origin)
    np.testing.assert_allclose(thresholds['shape_abs_p95'], np.quantile(np.abs(target.signed_target), .95))
    frame = diagnostic_frame(target.price_eur_mwh*.7+100, target.price_eur_mwh)
    shifted = diagnostic_frame(target.price_eur_mwh*.7+1100, target.price_eur_mwh+1000)
    first, second = stability_masks(frame, origin, thresholds), stability_masks(shifted, origin, thresholds)
    for name in ('SHAPE_HIGH','SHAPE_LOW','SHAPE_ABS_TAIL'):
        np.testing.assert_array_equal(first[name], second[name])
        assert first[name].sum()>0
    assert first['NEGATIVE_TRUTH'].sum()>0 and second['NEGATIVE_TRUTH'].sum()==0
    for month, part in frame.groupby(frame.index.tz_convert('Europe/Zurich').strftime('%Y-%m')):
        np.testing.assert_allclose(np.mean(part.full_error**2), np.mean(part.shape_error**2)+np.mean(part.level_error**2), atol=1e-10)


@pytest.mark.parametrize('kind', ['future', 'gap', 'wrong_target', 'nan'])
def test_training_thresholds_reject_invalid_history(kind):
    target = history()
    origin = pd.Timestamp('2024-04-01T00:00Z')
    if kind=='future': origin = target.index[-1]
    elif kind=='gap': target = target.drop(target.index[100])
    elif kind=='wrong_target': target.signed_target += 1
    elif kind=='nan': target.iloc[0,0] = np.nan
    with pytest.raises(ValueError): training_thresholds(target, origin)


def test_ramps_exclude_gaps_and_swiss_month_edges():
    target = history().price_eur_mwh.drop(history().index[100])
    frame = diagnostic_frame(target*.5, target)
    assert frame.truth_ramp.isna().sum()==3
    assert frame.ramp_error.isna().sum()==3


def screening_data():
    rows = []
    for candidate in ('signed-equal', 'blend'):
        for segment in ('ALL', 'SHAPE_LOW', 'SHAPE_HIGH', 'SHAPE_ABS_TAIL'):
            for field in ('shape_error', 'ramp_error'):
                value = 10. if candidate=='signed-equal' else 9.
                rows.append(dict(candidate=candidate, segment=segment, error_type=field, hours=1000, origins=4, mae=value, rmse=value))
    comparison = pd.DataFrame(rows)
    assessment = pd.concat([comparison.loc[comparison.segment.eq('ALL')].assign(origin=str(year)) for year in range(2023,2027)], ignore_index=True)
    return comparison, assessment


@pytest.mark.parametrize('failure', ['none', 'tail_regression', 'origin_regression', 'unsupported', 'gain'])
def test_screen_vetoes_regression_and_missing_tail_support(failure):
    comparison, assessment = screening_data()
    if failure=='tail_regression':
        comparison.loc[comparison.candidate.eq('blend')&comparison.segment.eq('SHAPE_LOW'), 'rmse'] = 11
    elif failure=='origin_regression':
        assessment.loc[assessment.candidate.eq('blend')&assessment.origin.eq('2023')&assessment.error_type.eq('ramp_error'), 'mae'] = 11
    elif failure=='unsupported':
        comparison.loc[comparison.segment.eq('SHAPE_HIGH'), ['hours', 'origins']] = [0, 0]
    elif failure=='gain':
        comparison.loc[comparison.candidate.eq('blend')&comparison.segment.eq('ALL'), 'mae'] = 9.9
    _, _, _, results = screen(comparison, assessment, ['blend'])
    assert results[0]['local_screen_pass'] == (failure=='none')
    assert results[0]['adoption'] is False
