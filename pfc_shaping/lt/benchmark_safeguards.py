"""Fail-closed local experiment checks; no model or operational authority."""
import numpy as np
import pandas as pd


class ForbiddenHourlyModel:
    """No fitted state: signed assembly must never consume hourly model input."""

    def apply(self, timestamps=None, calendar_df=None, reference_date=None, outages_forecast=None):
        raise RuntimeError("signed benchmark consumed forbidden hourly model")

    def __getattr__(self, name):
        raise RuntimeError(f"signed benchmark consumed forbidden hourly state: {name}")


def require_common_population(reference, observed, origin):
    """Exact ordered native hours imply identical Swiss months and maturities."""
    for index in (reference, observed):
        if (not isinstance(index, pd.DatetimeIndex) or index.tz is None
                or not index.is_unique or not index.is_monotonic_increasing or index.empty):
            raise ValueError("nonempty unique ordered aware population required")
    if origin.tzinfo is None or not reference.equals(observed):
        raise ValueError("candidate populations, hours and maturities differ")


def metric_ratio(value, reference, hours):
    if hours == 0:
        return None
    if not np.isfinite([value, reference]).all() or min(value, reference) < 0:
        raise ValueError("non-finite or negative metric in populated segment")
    if reference == 0:
        return 1. if value == 0 else None
    return float(value/reference)


def screen_segments(metrics, candidates, reference='signed-equal'):
    """Origin veto includes sparse adverse segments; robust support stays separate."""
    fields = ['shape_error', 'ramp_error']
    part = metrics.loc[metrics.error_type.isin(fields)]
    keys = ['origin', 'stage', 'segment', 'error_type']
    ref = part.loc[part.candidate.eq(reference)].set_index(keys)
    if ref.index.has_duplicates:
        raise ValueError('duplicate reference metrics')
    rows = []
    for candidate in candidates:
        frame = part.loc[part.candidate.eq(candidate)].set_index(keys)
        if not frame.index.equals(ref.index) or not frame.hours.equals(ref.hours):
            raise ValueError('metric populations differ')
        for key, row in frame.iterrows():
            r = ref.loc[key]
            a = metric_ratio(row.mae, r.mae, row.hours)
            b = metric_ratio(row.rmse, r.rmse, row.hours)
            adverse = bool(row.hours and (a is None or b is None or max(a, b) > 1.05))
            rows.append(dict(zip(keys, key), candidate=candidate, hours=int(row.hours),
                             mae_ratio=a, rmse_ratio=b, adverse=adverse))
    gates = pd.DataFrame(rows)
    decisions = []
    for candidate in candidates:
        g = gates.loc[gates.candidate.eq(candidate) & gates.stage.eq('final')].copy()
        support = g.groupby(['segment', 'error_type']).agg(hours=('hours', 'sum'),
            origins=('hours', lambda x: int((x > 0).sum())))
        unsupported = []
        regressions = []
        for row in g.itertuples():
            s = support.loc[(row.segment, row.error_type)]
            robust = s.hours >= 168 and s.origins >= 2
            if row.adverse:
                record = dict(origin=row.origin, segment=row.segment, error_type=row.error_type,
                              hours=row.hours, mae_ratio=row.mae_ratio, rmse_ratio=row.rmse_ratio)
                regressions.append(record)
                if not robust:
                    unsupported.append(record)
        allrows = metrics.loc[metrics.stage.eq('final') & metrics.segment.eq('ALL')
                             & metrics.error_type.eq('shape_error')]
        c = allrows.loc[allrows.candidate.eq(candidate)].set_index('origin')
        r = allrows.loc[allrows.candidate.eq(reference)].set_index('origin')
        gains = {m: 1-metric_ratio(c[m].mean(), r[m].mean(), c.hours.sum())
                 if r[m].mean() > 0 else 0. for m in ['mae', 'rmse']}
        wins = int((c.mae < r.mae).sum())
        tails = support.reindex(pd.MultiIndex.from_product(
            [['SHAPE_LOW', 'SHAPE_HIGH', 'SHAPE_ABS_TAIL'], fields]))
        tail_support = bool((tails.hours.ge(168) & tails.origins.ge(2)).all())
        decisions.append(dict(candidate=candidate, gains=gains, wins=wins,
            origin_regressions=regressions, unsupported_adverse_segments=unsupported,
            unsupported_adverse_count=len(unsupported), shape_tail_support=tail_support,
            local_screen_pass=bool(min(gains.values()) >= .02 and wins >= 3
                                   and not regressions and tail_support), adoption=False,
            evidence_class='EXPOSED_DEVELOPMENT_NO_HOLDOUT'))
    return gates, decisions
