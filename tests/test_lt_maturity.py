import numpy as np
import pandas as pd
import pytest

from scripts.lt_maturity_experiment import basis, pairs, fit, leads
from pfc_shaping.lt.signed_benchmark import closed_month_targets


def grid(start, end):
    return pd.date_range(start, end, freq='h', inclusive='left', tz='Europe/Zurich').tz_convert('UTC')


@pytest.fixture(scope='module')
def target():
    index = grid('2019-01-01', '2022-01-01')
    price = pd.Series(40+10*np.sin(np.arange(len(index))*2*np.pi/24), index=index)
    return closed_month_targets(price, pd.Timestamp('2022-01-01', tz='Europe/Zurich'))


@pytest.mark.parametrize('maturity,dimension', [(False, 54), (True, 108)])
def test_basis_neutral_dst(maturity, dimension):
    index = grid('2024-01-01', '2025-01-01')
    x = basis(index, pd.Timestamp('2023-12-31', tz='UTC'), maturity)
    assert x.shape == (8784, dimension)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(pd.DataFrame(x).groupby(index.tz_convert('Europe/Zurich').strftime('%Y-%m')).mean(), 0, atol=1e-14)


def test_maturity_varies_and_caps():
    index = grid('2024-01-01', '2028-01-01')
    origin = pd.Timestamp('2023-12-31', tz='UTC')
    x = basis(index, origin, True)
    np.testing.assert_allclose(x[:, 54:], x[:, :54]*np.minimum(leads(index, origin), 36)[:, None]/36, atol=1e-14)
    assert len(np.unique(leads(index, origin))) == 48


def test_pair_cutoffs_weights_and_leads(target):
    origin = pd.Timestamp('2022-01-01', tz='Europe/Zurich')
    frame = pairs(target, origin)
    assert (frame.history_end < frame.origin).all()
    assert (frame.origin < frame.delivery).all()
    assert (frame.delivery < origin).all()
    assert frame.lead.min() == 1 and frame.lead.max() > 12
    np.testing.assert_allclose(frame.groupby('delivery').weight.sum(), 1)
    assert len(frame) > frame.delivery.nunique()


def test_future_labels_rejected(target):
    with pytest.raises(ValueError, match='pre-outer-origin'):
        pairs(target, pd.Timestamp('2021-01-01', tz='UTC'))


def test_no_training_support_rejected(target):
    with pytest.raises(ValueError, match='no origin/delivery'):
        pairs(target.iloc[:100], pd.Timestamp('2022-01-01', tz='UTC'))


@pytest.mark.parametrize('maturity', [False, True])
def test_zero_residual_fit(target, maturity):
    frame = pairs(target, pd.Timestamp('2022-01-01', tz='Europe/Zurich'))
    frame['residual'] = 0.
    np.testing.assert_array_equal(fit(frame, maturity), 0)


def test_same_origin_month_rejected():
    with pytest.raises(ValueError, match='follow origin month'):
        basis(grid('2024-01-01', '2024-02-01'), pd.Timestamp('2024-01-01', tz='UTC'), True)


def test_duplicate_grid_rejected():
    index = grid('2024-01-01', '2024-02-01')
    with pytest.raises(ValueError, match='unique aware'):
        basis(index.append(index), pd.Timestamp('2023-12-01', tz='UTC'), True)
