"""Local EEX source diagnostics; never rewrites quotes or accepts conflicts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.validation.lt_benchmark_snapshots import utc
from pfc_shaping.validation.product_normalization import product_window_ch


def delivery_hours(product, load_type):
    start, end = product_window_ch(product)
    index = pd.date_range(start, end, freq='h', inclusive='left')
    peak = (index.dayofweek < 5) & (index.hour >= 8) & (index.hour < 20)
    if load_type == 'PEAK':
        return index[peak]
    if load_type == 'OFFPEAK':
        return index[~peak]
    if load_type != 'BASE':
        raise ValueError('unsupported load type')
    return index


def reconstruct_quote_conflicts(gates, surface, *, halfwidth=0.005, tolerance=1e-6):
    """Reconstruct parent-child identities from source quotes and Swiss hours.

    Half-cent compatibility is a declared rounding hypothesis, not a vendor
    precision claim, a correction, or a production acceptance tolerance.
    """
    if not np.isfinite([halfwidth, tolerance]).all() or halfwidth <= 0 or tolerance <= 0:
        raise ValueError('positive finite hypothesis/tolerance required')
    if surface.duplicated(['product', 'load_type']).any():
        raise ValueError('unique quote surface required')
    if not np.isfinite(surface.price.astype(float)).all():
        raise ValueError('finite source quotes required')
    quotes = surface.set_index(['product', 'load_type'])
    rows, direct = [], {}
    conflicts = gates.loc[gates.status.eq('QUOTE_CONFLICT')]
    for row in conflicts.loc[conflicts.load_type.isin(['BASE', 'PEAK'])].itertuples():
        key = (row.product, row.load_type)
        children = str(row.covered_by_quote_aware_products).split(',')
        parent_index = delivery_hours(*key)
        covered = parent_index[:0]
        weighted, child_receipts = 0., []
        for child in children:
            child_key = (child, row.load_type)
            if child == row.product or child_key not in quotes.index:
                raise ValueError('explicit finer source quotes required')
            index = delivery_hours(*child_key)
            if not index.isin(parent_index).all() or index.isin(covered).any():
                raise ValueError('children must partition parent without overlap')
            covered = covered.append(index)
            source = quotes.loc[child_key]
            passed = gates.loc[gates['product'].eq(child) & gates.load_type.eq(row.load_type)
                               & gates.gate_id.eq(row.gate_id)]
            if len(passed) != 1 or passed.status.iloc[0] != 'PASS':
                raise ValueError('each finer product must pass its hard gate')
            weight = len(index)/len(parent_index)
            weighted += weight*float(source.price)
            child_receipts.append(dict(product=child, hours=len(index), weight=weight, quote=float(source.price)))
        if not covered.sort_values().equals(parent_index):
            raise ValueError('incomplete child partition')
        parent = quotes.loc[key]
        gap = weighted-float(parent.price)
        if abs(gap-float(row.residual_eur_mwh)) > tolerance:
            raise ValueError('source identity does not reproduce delivered residual')
        result = dict(product=row.product, load_type=row.load_type, parent_quote=float(parent.price),
            reconstructed_mean=weighted, residual_eur_mwh=gap, hours=len(parent_index),
            delivered_residual=float(row.residual_eur_mwh), children=child_receipts,
            same_quote_date=all(pd.Timestamp(quotes.loc[(c, row.load_type), 'date']) == pd.Timestamp(parent.date) for c in children),
            parent_within_half_cent_of_fixed_children=abs(gap) <= halfwidth,
            parent_and_children_rounding_intervals_overlap=abs(gap) <= 2*halfwidth,
            classification='SOURCE_PARENT_CHILD_IDENTITY', accepted=False)
        rows.append(result)
        direct[key] = result
    for row in conflicts.loc[conflicts.load_type.eq('OFFPEAK')].itertuples():
        base, peak = direct[(row.product, 'BASE')], direct[(row.product, 'PEAK')]
        hb, hp = base['hours'], peak['hours']
        gap = (hb*base['residual_eur_mwh']-hp*peak['residual_eur_mwh'])/(hb-hp)
        if abs(gap-float(row.residual_eur_mwh)) > tolerance:
            raise ValueError('offpeak residual not implied by parent BASE/PEAK conflicts')
        rows.append(dict(product=row.product, load_type='OFFPEAK', residual_eur_mwh=gap,
            delivered_residual=float(row.residual_eur_mwh), hours=hb-hp,
            classification='DERIVED_BASE_PEAK_CONSEQUENCE', accepted=False))
    if len(rows) != len(conflicts):
        raise ValueError('unhandled conflict identity')
    return dict(rows=rows, direct_conflicts=len(direct), derived_conflicts=len(rows)-len(direct),
        hypothesis_halfwidth_eur_mwh=halfwidth, vendor_rounding_confirmed=False,
        accepted_conflicts=0, authority=dict(AUTHORITIES))


def assess_eex_freshness(surface, *, observed_at, valuation_at, expected_quote_date):
    """Expected publication date is supplied explicitly; no weekday calendar guess."""
    observed, valuation = utc(observed_at), utc(valuation_at)
    expected = pd.Timestamp(expected_quote_date)
    if expected.tzinfo is not None or expected != expected.normalize() or pd.isna(expected):
        raise ValueError('explicit calendar quotation date required')
    if expected.date() >= valuation.tz_convert('Europe/Zurich').date() or observed > valuation:
        raise ValueError('closed quotation date and pre-valuation observation required')
    if surface.empty or surface.duplicated(['product','load_type']).any():
        raise ValueError('nonempty unique quote surface required')
    rows = []
    for row in surface.itertuples():
        quote_date = pd.Timestamp(row.date)
        loaded = utc(row.fact_load_timestamp_utc)
        status = ('POST_OBSERVATION_LOAD' if loaded > observed else
                  'UNEXPECTED_QUOTE_DATE' if quote_date > expected else
                  'STALE_QUOTE' if quote_date < expected else 'EXPECTED_DATE_OBSERVED')
        if row.market != 'CH' or str(row.unit).upper() != 'EUR/MWH' or not np.isfinite(float(row.price)):
            status = 'INVALID_MARKET_UNIT_VALUE'
        start, end = product_window_ch(row.product)
        if start <= valuation.tz_convert('Europe/Zurich'):
            status = 'DELIVERY_ALREADY_STARTED'
        rows.append(dict(product=row.product, load_type=row.load_type, quotation_date=quote_date.date().isoformat(),
            expected_quote_date=expected.date().isoformat(), fact_load_at_utc=loaded.isoformat(),
            observed_at_utc=observed.isoformat(), load_to_observation_hours=(observed-loaded).total_seconds()/3600,
            delivery_year=start.year, delivery_start=start.isoformat(), delivery_end_exclusive=end.isoformat(),
            status=status, independent_availability_proven=False))
    return pd.DataFrame(rows)
