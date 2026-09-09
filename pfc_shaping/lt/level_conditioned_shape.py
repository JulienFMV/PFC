"""Experimental additive slope against archived solver levels, never truth levels."""
import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index


def design(index, levels):
    if (index.tz is None or not index.is_unique or not index.is_monotonic_increasing
            or not isinstance(levels, pd.Series) or not levels.index.equals(index)
            or not np.isfinite(levels).all()):
        raise ValueError('aligned finite solver levels and unique aware hours required')
    month = index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    if levels.groupby(month).nunique().max() != 1:
        raise ValueError('solver level must be constant within each month')
    cal = enrich_15min_index(index)
    h = cal.heure_hce.to_numpy(dtype=float)
    harmonic = np.column_stack([f(2*np.pi*k*h/24) for k in (1, 2, 3) for f in (np.sin, np.cos)])
    x = np.column_stack([harmonic*(cal[field].to_numpy() == value)[:, None]
        for field, values in [('saison', ('Hiver', 'Printemps', 'Ete', 'Automne')),
                             ('type_jour', ('Ouvrable', 'Samedi', 'Dimanche', 'Ferie_CH', 'Ferie_DE'))]
        for value in values])
    x *= levels.to_numpy()[:, None]/100.
    return x-pd.DataFrame(x).groupby(month).transform('mean').to_numpy()


def fit(pairs, outer_origin, ridge):
    if not np.isfinite(ridge) or ridge <= 0 or outer_origin.tzinfo is None:
        raise ValueError('positive fixed ridge and aware outer origin required')
    if pairs.empty:
        return np.zeros(54)
    if (not (pairs.origin < pairs.delivery).all() or not (pairs.delivery < outer_origin).all()
            or not (pairs.origin < outer_origin).all()
            or pairs.duplicated(['origin', 'delivery']).any()
            or not np.isfinite(pairs[['level', 'residual', 'weight']]).all().all()
            or not pairs.weight.gt(0).all()):
        raise ValueError('causal finite unique origin/delivery pairs required')
    months = pairs.delivery.dt.tz_convert('Europe/Zurich').dt.tz_localize(None).dt.to_period('M')
    ends = (months+1).dt.start_time.dt.tz_localize('Europe/Zurich')
    if not (ends <= outer_origin).all():
        raise ValueError('training delivery months must be closed before origin')
    gram, rhs = np.zeros((54, 54)), np.zeros(54)
    for _, group in pairs.groupby('origin'):
        index = pd.DatetimeIndex(group.delivery)
        x = design(index, pd.Series(group.level.to_numpy(), index=index))
        w = group.weight.to_numpy()
        gram += x.T@(w[:, None]*x)
        rhs += x.T@(w*group.residual.to_numpy())
    mass = pairs.weight.sum()
    return np.linalg.solve(gram/mass+ridge*np.eye(54), rhs/mass)
