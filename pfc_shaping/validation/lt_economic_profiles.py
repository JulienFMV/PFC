"""Explicit fixed-profile EUR valuation diagnostics, without economic admission.

No assumed FMV product population, capture premium, BLOC13 payoff or hydro policy.
Volumes are interval MWh, so quarter-hour values are not multiplied by four.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.validation.lt_benchmark_snapshots import utc


def value_fixed_profile(prices, profile, *, valuation_at, native_interval_minutes):
    required = {'timestamp_utc','volume_mwh','direction','profile_id','profile_version',
                'available_at_utc','source_document_id'}
    if set(profile.columns) != required or profile.empty or native_interval_minutes not in (15, 60):
        raise ValueError('exact versioned fixed-profile schema and native interval required')
    idx = pd.DatetimeIndex([utc(v) for v in profile.timestamp_utc])
    if (not idx.is_unique or not idx.is_monotonic_increasing or not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.tz is None or not prices.index.equals(idx)):
        raise ValueError('exact ordered common native grid required')
    delta = pd.Timedelta(minutes=native_interval_minutes)
    if len(idx) > 1 and not (idx[1:]-idx[:-1] == delta).all():
        raise ValueError('no missing profile intervals or implicit filling')
    available = pd.DatetimeIndex([utc(v) for v in profile.available_at_utc])
    if (available > utc(valuation_at)).any() or (idx < utc(valuation_at)).any():
        raise ValueError('profile must be available at valuation for future delivery')
    for field in ['profile_id','profile_version','source_document_id']:
        if profile[field].isna().any() or profile[field].astype(str).str.strip().eq('').any():
            raise ValueError('profile identity and lineage required')
    if profile.profile_id.nunique() != 1 or profile.profile_version.nunique() != 1:
        raise ValueError('one explicit versioned population per comparison')
    if profile.direction.nunique() != 1 or profile.direction.iloc[0] not in ('GENERATION','CONSUMPTION'):
        raise ValueError('one explicit generation/consumption direction required')
    volume, price = profile.volume_mwh.to_numpy(dtype=float), prices.to_numpy(dtype=float)
    if not np.isfinite(volume).all() or (volume < 0).any() or not np.isfinite(price).all():
        raise ValueError('finite prices and finite nonnegative MWh required')
    total = float(volume.sum())
    if not np.isfinite(total):
        raise ValueError('nonfinite total volume')
    result = dict(profile_id=str(profile.profile_id.iloc[0]), profile_version=str(profile.profile_version.iloc[0]),
        direction=str(profile.direction.iloc[0]), rows=len(profile), total_mwh=total, currency='EUR',
        native_interval_minutes=native_interval_minutes, authority=dict(AUTHORITIES))
    if total == 0:
        return dict(result, status='UNSUPPORTED_ZERO_VOLUME', capture_price_eur_mwh=None, signed_cashflow_eur=None)
    unsigned = float(np.dot(price, volume))
    if not np.isfinite(unsigned):
        raise ValueError('nonfinite cashflow')
    sign = 1 if profile.direction.iloc[0] == 'GENERATION' else -1
    return dict(result, status='LOCAL_VALUATION_NOT_ECONOMIC_ADMISSION', capture_price_eur_mwh=unsigned/total,
        signed_cashflow_eur=sign*unsigned, negative_price_mwh=float(volume[price < 0].sum()))
