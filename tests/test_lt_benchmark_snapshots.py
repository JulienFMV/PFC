import copy
import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.validation.lt_benchmark_snapshots import append_snapshot, read_registry, complete_months, score_closed_months


@pytest.fixture
def pilot(tmp_path):
    build = tmp_path/'build'
    build.mkdir()
    artifact = build/'artifact.bin'
    artifact.write_bytes(b'fixed evidence')
    ref = dict(path='build/artifact.bin', sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    registry = build/'registry'
    registry.mkdir()
    entry = dict(schema='fmv-benchmark-snapshot.v1', valuation_at_utc='2026-09-08T10:00:00Z',
        candidate_committed_at_utc='2026-09-08T10:01:00Z', registered_at_utc='2026-09-08T10:02:00Z',
        recipe=ref.copy(), candidate=ref.copy(), authority=dict(AUTHORITIES),
        evidence_class='LOCAL_OBSERVED_NOT_INDEPENDENTLY_AUTHENTICATED',
        inputs={role:dict(artifact=ref.copy(), observed_at_utc='2026-09-08T09:00:00Z', issue_at_utc=None,
                         vendor_availability_authenticated=False) for role in ['EEX','CH_HISTORY','OMPEX','LSEG']})
    return tmp_path, registry, entry


def test_append_and_verify(pilot):
    root, registry, entry = pilot
    path = append_snapshot(root, registry, entry, now='2026-09-08T11:00Z')
    assert path.name == '0001-2026-09-08.json'
    records, digest = read_registry(root, registry, now='2026-09-08T11:00Z')
    assert len(records) == 1 and digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_preserves_v1_history_and_binds_new_eex_surface(pilot):
    root, registry, entry = pilot
    first = append_snapshot(root, registry, entry, now='2026-09-08T11:00Z')
    first_bytes = first.read_bytes()
    second = copy.deepcopy(entry)
    second['schema'] = 'fmv-benchmark-snapshot.v2'
    for key in ['valuation_at_utc', 'candidate_committed_at_utc', 'registered_at_utc']:
        second[key] = second[key].replace('2026-09-08', '2026-09-09')
    eex = root/'build/eex.parquet'
    pd.DataFrame(dict(date=pd.to_datetime(['2026-09-08']), price=[50.])).to_parquet(eex)
    source = second['inputs']['EEX']
    source['artifact'] = dict(path='build/eex.parquet', sha256=hashlib.sha256(eex.read_bytes()).hexdigest())
    source['observed_at_utc'] = '2026-09-09T09:00:00Z'
    source['quotation_dates'] = dict.fromkeys(['raw_latest_date', 'normalized_latest_date', 'merged_latest_date'], '2026-09-08')
    append_snapshot(root, registry, second, now='2026-09-09T11:00Z')
    records, _ = read_registry(root, registry, now='2026-09-09T11:00Z')
    assert len(records) == 2 and first.read_bytes() == first_bytes
    assert records[1]['entry']['inputs']['EEX']['quotation_dates']['merged_latest_date'] == '2026-09-08'


@pytest.mark.parametrize('change', ['after_cutoff','future_registration','late_issue','authority','missing_source','naive','changed_hash'])
def test_bad_entry_rejected_without_write(pilot, change):
    root, registry, entry = pilot
    if change == 'after_cutoff': entry['inputs']['EEX']['observed_at_utc'] = '2026-09-08T10:05Z'
    if change == 'future_registration': entry['registered_at_utc'] = '2026-09-10T12:00Z'
    if change == 'late_issue': entry['inputs']['LSEG']['issue_at_utc'] = '2026-09-08T10:05Z'
    if change == 'authority': entry['authority']['production'] = True
    if change == 'missing_source': del entry['inputs']['OMPEX']
    if change == 'naive': entry['valuation_at_utc'] = '2026-09-08 10:00'
    if change == 'changed_hash': entry['candidate']['sha256'] = '0'*64
    with pytest.raises(ValueError): append_snapshot(root, registry, entry, now='2026-09-08T11:00Z')
    assert not list(registry.iterdir())


def test_same_day_rejected(pilot):
    root, registry, entry = pilot
    append_snapshot(root, registry, entry, now='2026-09-08T11:00Z')
    entry['valuation_at_utc'] = '2026-09-08T10:00:30Z'
    with pytest.raises(ValueError, match='later Swiss day'):
        append_snapshot(root, registry, entry, now='2026-09-08T11:00Z')


def test_chain_tampering_rejected(pilot):
    root, registry, entry = pilot
    path = append_snapshot(root, registry, entry, now='2026-09-08T11:00Z')
    content = json.loads(path.read_text())
    content['previous_sha256'] = '0'*64
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='chain'): read_registry(root, registry, now='2026-09-08T11:00Z')


def test_outside_build_rejected(pilot):
    root, registry, entry = pilot
    # Synthetic unsafe target remains inside the test basetemp, never an external real path.
    outside = root/'outside.bin'
    outside.write_bytes(b'fixed evidence')
    entry['candidate']['path'] = 'outside.bin'
    with pytest.raises(ValueError, match='below build'):
        append_snapshot(root, registry, entry, now='2026-09-08T11:00Z')


@pytest.mark.parametrize('month,hours', [('2026-03',743), ('2026-10',745)])
def test_dst_complete_month(month, hours):
    start = pd.Timestamp(month+'-01', tz='Europe/Zurich')
    end = (pd.Period(month, freq='M')+1).start_time.tz_localize('Europe/Zurich')
    index = pd.date_range(start, end, freq='h', inclusive='left').tz_convert('UTC')
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    assert len(complete_months(series)) == hours
    with pytest.raises(ValueError, match='complete Swiss'): complete_months(series.iloc[1:])


def test_future_truth_rejected():
    index = pd.date_range('2026-10-01', '2026-11-01', freq='h', inclusive='left', tz='Europe/Zurich').tz_convert('UTC')
    truth = pd.Series(20+np.sin(np.arange(len(index))), index=index)
    with pytest.raises(ValueError, match='future/unavailable'):
        score_closed_months(truth, truth, truth_available_at='2026-11-02T00:00Z', now='2026-10-20T00:00Z', committed_at='2026-09-08T12:00Z')


def test_monthly_level_score_separate():
    index = pd.date_range('2026-10-01', '2026-11-01', freq='h', inclusive='left', tz='Europe/Zurich').tz_convert('UTC')
    truth = pd.Series(20+np.sin(np.arange(len(index))), index=index)
    result = score_closed_months(truth+5, truth, truth_available_at='2026-11-02T00:00Z', now='2026-11-03T00:00Z', committed_at='2026-09-08T12:00Z')
    assert result['status'] == 'LOCAL_DIAGNOSTIC_NOT_SCIENTIFIC_ADMISSION'
    assert not any(result['authority'].values())


def test_second_day_and_cap(pilot):
    root, registry, entry = pilot
    for offset in range(20):
        day = pd.Timestamp('2026-09-08T10:00Z')+pd.Timedelta(days=offset)
        e = copy.deepcopy(entry)
        for field, minutes in [('valuation_at_utc',0),('candidate_committed_at_utc',1),('registered_at_utc',2)]:
            e[field] = (day+pd.Timedelta(minutes=minutes)).isoformat()
        append_snapshot(root, registry, e, now=day+pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match='pilot cap'):
        append_snapshot(root, registry, e, now=day+pd.Timedelta(hours=1))
    assert len(read_registry(root, registry, now=day+pd.Timedelta(hours=1))[0]) == 20
