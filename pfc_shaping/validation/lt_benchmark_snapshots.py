"""Local daily benchmark registry; hash continuity conveys no external authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.path_safety import assert_absolute_path_has_no_links
from pfc_shaping.validation.ch_lt_prospective_hourly_scoring import score_hourly_prediction


def utc(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or pd.isna(timestamp):
        raise ValueError('explicit timezone required')
    return timestamp.tz_convert('UTC')


def bound_file(root, reference):
    if set(reference) != {'path', 'sha256'}:
        raise ValueError('exact artifact path/hash required')
    raw = root/reference['path']
    path = raw.resolve(strict=True)
    if not path.is_relative_to(root/'build') or not path.is_file():
        raise ValueError('artifact must remain below build')
    assert_absolute_path_has_no_links(raw)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != reference['sha256']:
        raise ValueError('artifact hash mismatch')
    return path


def validate_entry(root, entry, now):
    required = {'schema', 'valuation_at_utc', 'candidate_committed_at_utc', 'registered_at_utc',
                'recipe', 'candidate', 'inputs', 'authority', 'evidence_class'}
    if set(entry) != required or entry['schema'] != 'fmv-benchmark-snapshot.v1':
        raise ValueError('unexpected snapshot schema')
    if (set(entry['authority']) != set(AUTHORITIES) or any(v is not False for v in entry['authority'].values())
            or entry['evidence_class'] != 'LOCAL_OBSERVED_NOT_INDEPENDENTLY_AUTHENTICATED'):
        raise ValueError('all authorities must remain false; local evidence only')
    valuation, committed, registered = [utc(entry[k]) for k in
        ['valuation_at_utc', 'candidate_committed_at_utc', 'registered_at_utc']]
    if not valuation <= committed <= registered <= utc(now):
        raise ValueError('invalid capture/commit/registration chronology')
    bound_file(root, entry['recipe'])
    bound_file(root, entry['candidate'])
    if set(entry['inputs']) != {'EEX', 'CH_HISTORY', 'OMPEX', 'LSEG'}:
        raise ValueError('all four source roles required')
    for source in entry['inputs'].values():
        if set(source) != {'artifact', 'observed_at_utc', 'issue_at_utc', 'vendor_availability_authenticated'}:
            raise ValueError('exact source observation schema required')
        observed = utc(source['observed_at_utc'])
        if observed > valuation:
            raise ValueError('source observed after valuation')
        if source['issue_at_utc'] is not None and utc(source['issue_at_utc']) > observed:
            raise ValueError('issue after observation')
        if source['vendor_availability_authenticated'] is not False:
            raise ValueError('this local registry cannot assert vendor authentication')
        bound_file(root, source['artifact'])
    return valuation.tz_convert('Europe/Zurich').strftime('%Y-%m-%d')


def read_registry(root, registry, *, now):
    root, registry = Path(root).resolve(strict=True), Path(registry).resolve(strict=True)
    if not registry.is_relative_to(root/'build'):
        raise ValueError('registry must remain below build')
    assert_absolute_path_has_no_links(registry)
    records, previous, days = [], None, set()
    for sequence, path in enumerate(sorted(registry.glob('*.json')), start=1):
        record = json.loads(path.read_text(encoding='utf-8'))
        if set(record) != {'sequence', 'previous_sha256', 'entry'} or record['sequence'] != sequence or record['previous_sha256'] != previous:
            raise ValueError('broken registry sequence/hash chain')
        day = validate_entry(root, record['entry'], now)
        if path.name != f'{sequence:04d}-{day}.json' or day in days:
            raise ValueError('duplicate or invalid daily record')
        if records and (utc(record['entry']['valuation_at_utc']) <= utc(records[-1]['entry']['valuation_at_utc'])
                        or record['entry']['recipe'] != records[0]['entry']['recipe']):
            raise ValueError('strictly increasing valuations and frozen recipe required')
        days.add(day)
        records.append(record)
        previous = hashlib.sha256(path.read_bytes()).hexdigest()
    return records, previous


def append_snapshot(root, registry, entry, *, now):
    root, registry = Path(root).resolve(strict=True), Path(registry).resolve(strict=True)
    day = validate_entry(root, entry, now)
    records, previous = read_registry(root, registry, now=now)
    if len(records) >= 20:
        raise ValueError('twenty-snapshot pilot cap reached')
    if records:
        prior = records[-1]['entry']
        if day <= utc(prior['valuation_at_utc']).tz_convert('Europe/Zurich').strftime('%Y-%m-%d'):
            raise ValueError('one genuinely later Swiss day per snapshot')
        if entry['recipe'] != records[0]['entry']['recipe']:
            raise ValueError('frozen recipe changed')
    record = dict(sequence=len(records)+1, previous_sha256=previous, entry=entry)
    path = registry/f'{len(records)+1:04d}-{day}.json'
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def complete_months(series):
    index = series.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or not index.is_unique
            or not index.is_monotonic_increasing or not np.isfinite(series.to_numpy(dtype=float)).all() or series.empty):
        raise ValueError('nonempty ordered finite unique hourly series required')
    groups = index.tz_convert('Europe/Zurich').strftime('%Y-%m')
    for month, part in series.groupby(groups):
        begin = pd.Timestamp(month+'-01', tz='Europe/Zurich')
        end = (pd.Period(month, freq='M')+1).start_time.tz_localize('Europe/Zurich')
        if not part.index.equals(pd.date_range(begin, end, freq='h', inclusive='left').tz_convert('UTC')):
            raise ValueError('complete Swiss months required; no gap filling')
    return groups


def score_closed_months(prediction, truth, *, truth_available_at, now, committed_at):
    """Local numeric diagnostics only, even with a claimed finalization receipt."""
    groups = complete_months(prediction)
    complete_months(truth)
    if not prediction.index.equals(truth.index):
        raise ValueError('exact common grid required')
    end = prediction.index[-1]+pd.Timedelta(hours=1)
    if not utc(committed_at) < prediction.index[0] or not end <= utc(truth_available_at) <= utc(now):
        raise ValueError('future/unavailable truth or late prediction commitment')
    rows = []
    for month in sorted(set(groups)):
        mask = groups == month
        rows.append(dict(month=month, diagnostics=score_hourly_prediction(prediction.loc[mask], truth.loc[mask], label=month)))
    return dict(status='LOCAL_DIAGNOSTIC_NOT_SCIENTIFIC_ADMISSION', months=rows, authority=dict(AUTHORITIES))
