import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.benchmark_safeguards import (
    ForbiddenHourlyModel, require_common_population, screen_segments,
)
from pfc_shaping.lt.evaluation_curve_assembly import _validated_prices, CurveAssemblyError
from tests.test_signed_hourly_assembly import fixture
from scripts.run_lt_signed_composition import assembler


def test_missing_month_rejected_despite_parent_and_previous_year():
    index = pd.date_range('2024-01-01', '2024-03-01', freq='h', tz='Europe/Zurich')
    with pytest.raises(CurveAssemblyError, match='every delivery month'):
        _validated_prices({'2024-01': 40, '2024-Q1': 50, '2023-02': 60}, index)


def test_complete_month_keys_keep_negative_and_zero_levels():
    index = pd.date_range('2024-02-01', '2024-03-31', freq='h', tz='Europe/Zurich')
    values = {'2024-02': -10., '2024-03': 0.}
    assert _validated_prices(values, index) == values


@pytest.mark.parametrize('month', ['2024-02', '2024-03', '2024-10'])
def test_real_signed_assembler_cannot_consume_any_hourly_state(month):
    _, kwargs = fixture(month)
    assembly = assembler(ForbiddenHourlyModel())
    result = assembly.build(**kwargs)
    assert result.f_H.eq(1).all()
    with pytest.raises(RuntimeError, match='forbidden'):
        assembly.sh.apply()
    with pytest.raises(RuntimeError, match='forbidden'):
        _ = assembly.sh.f_W_


@pytest.mark.parametrize('change', ['drop', 'shift', 'reverse', 'duplicate'])
def test_population_mismatch_rejected(change):
    index = pd.date_range('2024-01-01', periods=200, freq='h', tz='UTC')
    other = {'drop': index[:-1], 'shift': index+pd.Timedelta(hours=1),
             'reverse': index[::-1], 'duplicate': index.append(index[:1])}[change]
    with pytest.raises(ValueError):
        require_common_population(index, other, pd.Timestamp('2023-12-01', tz='UTC'))


def metrics():
    return pd.DataFrame([dict(origin='2023', stage='final', segment=s,
        error_type=e, candidate=c, hours=200, mae=v, rmse=v)
        for s in ['ALL', 'YEAR_2023', 'SHAPE_LOW', 'SHAPE_HIGH', 'SHAPE_ABS_TAIL']
        for e in ['shape_error', 'ramp_error'] for c, v in [('signed-equal', 1.), ('candidate', 1.1)]])


def test_single_origin_adverse_segment_visible_and_vetoes():
    _, decisions = screen_segments(metrics(), ['candidate'])
    d = decisions[0]
    assert d['unsupported_adverse_count'] == 10
    assert d['origin_regressions'] and not d['local_screen_pass']
    assert not d['shape_tail_support']


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_populated_metric_rejected(value):
    frame = metrics()
    frame.loc[1, 'mae'] = value
    with pytest.raises(ValueError, match='non-finite'):
        screen_segments(frame, ['candidate'])


def test_metric_population_mismatch_rejected():
    with pytest.raises(ValueError, match='populations'):
        screen_segments(metrics().iloc[:-1], ['candidate'])
