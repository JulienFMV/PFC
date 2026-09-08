"""Bounded retrospective origin/delivery residual experiment; no authority."""
import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.signed_benchmark import calendar_cell_reference
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape

RIDGE = .1
SEASONS = ('Hiver', 'Printemps', 'Ete', 'Automne')
DAYS = ('Ouvrable', 'Samedi', 'Dimanche', 'Ferie_CH', 'Ferie_DE')


def leads(index, origin):
    local = index.tz_convert('Europe/Zurich')
    o = origin.tz_convert('Europe/Zurich')
    return np.asarray(local.year*12+local.month-(o.year*12+o.month), dtype=int)


def basis(index, origin, maturity):
    if index.tz is None or origin.tzinfo is None or not index.is_unique:
        raise ValueError('unique aware delivery and origin required')
    lead = leads(index, origin)
    if np.any(lead < 1):
        raise ValueError('delivery must follow origin month')
    cal = enrich_15min_index(index)
    hour = cal.heure_hce.to_numpy(dtype=float)
    harmonics = np.column_stack([f(2*np.pi*k*hour/24) for k in (1, 2, 3) for f in (np.sin, np.cos)])
    columns = [harmonics*(cal[column].to_numpy() == value)[:, None]
               for column, values in [('saison', SEASONS), ('type_jour', DAYS)] for value in values]
    x = np.column_stack(columns)
    if maturity:
        x = np.column_stack([x, x*np.minimum(lead, 36)[:, None]/36])
    # Learn the same monthly-neutral function that is applied at inference.
    months = index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    return x-pd.DataFrame(x).groupby(months).transform('mean').to_numpy()


def pairs(target, outer_origin):
    if target.empty or target.index.max() >= outer_origin:
        raise ValueError('strict pre-outer-origin labels required')
    months = target.index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    origins = pd.date_range(target.index.min().tz_convert('Europe/Zurich').normalize(),
                            outer_origin.tz_convert('Europe/Zurich'), freq='QS')
    parts = []
    for start in origins:
        origin = (start-pd.Timedelta(hours=12)).tz_convert('UTC')
        closed = months < origin.tz_convert('Europe/Zurich').strftime('%Y-%m')
        history = target.loc[closed]
        if len(set(months[closed])) < 12:
            continue
        lead = leads(target.index, origin)
        future = target.loc[(lead >= 1) & (lead <= 36)]
        if future.empty:
            continue
        raw, _ = calendar_cell_reference(history.signed_target, future.index)
        baseline = center_signed_hourly_shape(pd.Series(raw, index=future.index))
        parts.append(pd.DataFrame(dict(delivery=future.index, origin=origin,
            history_end=history.index[-1], lead=leads(future.index, origin),
            baseline=baseline.to_numpy(), residual=future.signed_target.to_numpy()-baseline.to_numpy())))
    if not parts:
        raise ValueError('no origin/delivery training pairs')
    result = pd.concat(parts, ignore_index=True)
    result['weight'] = 1/result.groupby('delivery').delivery.transform('size')
    return result


def fit(pair_frame, maturity):
    dimension = 108 if maturity else 54
    gram = np.zeros((dimension, dimension))
    rhs = np.zeros(dimension)
    for origin, part in pair_frame.groupby('origin'):
        x = basis(pd.DatetimeIndex(part.delivery), origin, maturity)
        w = part.weight.to_numpy()
        gram += x.T@(w[:, None]*x)
        rhs += x.T@(w*part.residual.to_numpy())
    mass = pair_frame.weight.sum()
    return np.linalg.solve(gram/mass+RIDGE*np.eye(dimension), rhs/mass)
