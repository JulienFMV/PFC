import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.level_conditioned_shape import design, fit


def data():
    idx = pd.date_range('2024-03-01', '2024-04-01', freq='h', inclusive='left', tz='Europe/Zurich').tz_convert('UTC')
    return idx, pd.Series(40., index=idx)


@pytest.mark.parametrize('level', [-50., 0., .001, 100.])
def test_signed_level_design_monthly_neutral(level):
    index, levels = data()
    levels[:] = level
    x = design(index, levels)
    assert x.shape == (743, 54)
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-14)
    assert np.isfinite(x).all()


def test_level_not_constant_rejected():
    index, levels = data()
    levels.iloc[0] = 80
    with pytest.raises(ValueError, match='constant'):
        design(index, levels)


def test_fit_matches_independent_augmented_least_squares():
    index, levels = data()
    x = design(index, levels)
    y = x[:, 0]*3
    pair = pd.DataFrame(dict(origin=pd.Timestamp('2024-02-01', tz='UTC'), delivery=index,
                             level=levels.to_numpy(), residual=y, weight=1.))
    beta = fit(pair, pd.Timestamp('2024-04-01', tz='Europe/Zurich'), 1.)
    expected = np.linalg.lstsq(np.vstack([x/np.sqrt(len(x)), np.eye(54)]),
                              np.r_[y/np.sqrt(len(x)), np.zeros(54)], rcond=None)[0]
    np.testing.assert_allclose(beta, expected, atol=1e-12)
    with pytest.raises(ValueError, match='causal'):
        fit(pair, pd.Timestamp('2024-03-15', tz='UTC'), 1.)
    with pytest.raises(ValueError, match='closed'):
        fit(pair.iloc[:10], pd.Timestamp('2024-03-20', tz='UTC'), 1.)


def test_no_pairs_is_explicit_zero_ablation():
    np.testing.assert_array_equal(fit(pd.DataFrame(), pd.Timestamp('2024-01-01', tz='UTC'), 10), np.zeros(54))
