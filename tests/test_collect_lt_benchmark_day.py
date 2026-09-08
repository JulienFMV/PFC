import json

import pandas as pd
import pytest

from scripts.collect_lt_benchmark_day import BoundedQueries, due_day, normalize_lseg, fresh_output, capture_lock, merge_observed_eex_dates


def response(rows=None, *, state='SUCCEEDED', truncated=False):
    data = [['1']] if rows is None else rows
    return dict(statement_id='s1',status=dict(state=state),
        manifest=dict(truncated=truncated,total_row_count=len(data),schema=dict(columns=[dict(name='value')])),
        result=dict(data_array=data))


def test_query_pagination_and_budget(tmp_path):
    calls=[]
    def request(method,path,body=None):
        calls.append((method,path,body))
        result=response()
        result['manifest']['total_row_count']=2
        result['result']['next_chunk_internal_link']='/api/2.0/sql/statements/s1/result/chunks/1'
        return dict(data_array=[['2']]) if method=='GET' else result
    q=BoundedQueries(request,'test',tmp_path)
    assert q.query('read','SELECT value FROM test',row_limit=3).value.tolist()==['1','2']
    q.count=6
    with pytest.raises(ValueError,match='budget'): q.query('over','SELECT value FROM test',row_limit=3)
    assert len(calls)==2


@pytest.mark.parametrize('kind',['truncated','sentinel','row_mismatch','external_chunk','failed'])
def test_query_rejects_incomplete_or_failed_result(tmp_path,kind):
    r=response(truncated=kind=='truncated',state='FAILED' if kind=='failed' else 'SUCCEEDED')
    if kind=='row_mismatch': r['manifest']['total_row_count']=2
    if kind=='external_chunk': r['result']['next_chunk_internal_link']='https://example.invalid/chunk'
    q=BoundedQueries(lambda *args:r,'test',tmp_path)
    with pytest.raises(ValueError): q.query('read','SELECT value FROM test',row_limit=1 if kind=='sentinel' else 3)


def test_timeout_cancels_own_statement(tmp_path):
    times=iter([0,181])
    calls=[]
    def request(method,path,body=None):
        calls.append(path)
        return response(state='RUNNING')
    q=BoundedQueries(request,'test',tmp_path,monotonic=lambda:next(times),sleep=lambda _:None)
    with pytest.raises(TimeoutError): q.query('read','SELECT value FROM test',row_limit=3)
    assert calls[-1]=='/api/2.0/sql/statements/s1/cancel'


@pytest.mark.parametrize('auto_stop',[0,46,None])
def test_no_start_without_bounded_autostop(tmp_path,auto_stop):
    calls=[]
    def request(method,path,body=None):
        calls.append(method)
        return dict(state='STOPPED',auto_stop_mins=auto_stop)
    q=BoundedQueries(request,'test',tmp_path)
    with pytest.raises(ValueError,match='auto-stop'): q.ready()
    assert calls==['GET']


def test_authorized_start_once_no_forbidden_stop_retry(tmp_path):
    calls=[]
    states=iter(['STOPPED','RUNNING','RUNNING'])
    def request(method,path,body=None):
        calls.append((method,path))
        return dict(state=next(states),auto_stop_mins=45) if method=='GET' else {}
    q=BoundedQueries(request,'test',tmp_path,sleep=lambda _:None)
    q.ready()
    q.finish()
    assert sum(path.endswith('/start') for _,path in calls)==1
    assert not any(path.endswith('/stop') for _,path in calls)
    assert json.loads((tmp_path/'warehouse-final.json').read_text())['shutdown_confirmed'] is False


def lseg_fixture():
    idx=pd.date_range('2026-10-01',periods=3,freq='h',tz='UTC')
    return pd.DataFrame(dict(ValueStartDateTimeUtc=idx,DateTimeUtc=idx+pd.Timedelta(hours=1),
        Value=[-1,0,1],Unit='EUR/MWh',ForecastIssuedAtUtc='2026-09-08T00:00Z',
        PipelineFirstSeenAtUtc='2026-09-08T07:00Z',PullTimestampUtc='2026-09-08T07:00Z'))


def test_lseg_preserves_negative_values():
    series,issue=normalize_lseg(lseg_fixture(),pd.Timestamp('2026-09-08T10:00Z'))
    assert series.tolist()==[-1,0,1]
    assert issue.startswith('2026-09-08')


@pytest.mark.parametrize('kind',['mixed_issue','late','gap','duplicate','wrong_hour'])
def test_lseg_rejects_bad_vintage_or_hour_grid(kind):
    frame=lseg_fixture()
    if kind=='mixed_issue': frame.loc[0,'ForecastIssuedAtUtc']='2026-09-07T00:00Z'
    if kind=='late': frame.loc[0,'PullTimestampUtc']='2026-09-09T07:00Z'
    if kind=='gap': frame=frame.drop(1)
    if kind=='duplicate': frame=pd.concat([frame,frame.iloc[:1]])
    if kind=='wrong_hour': frame.loc[0,'DateTimeUtc']=frame.loc[0,'ValueStartDateTimeUtc']
    with pytest.raises(ValueError): normalize_lseg(frame,pd.Timestamp('2026-09-08T10:00Z'))


def test_same_day_due_guard_calls_only_registry(monkeypatch,tmp_path):
    record=dict(entry=dict(valuation_at_utc='2026-09-08T09:00Z'))
    monkeypatch.setattr('scripts.collect_lt_benchmark_day.read_registry',lambda *a,**k:([record],None))
    assert due_day(tmp_path,tmp_path,pd.Timestamp('2026-09-08T20:00Z'))[0]=='ALREADY_CAPTURED_TODAY'
    assert due_day(tmp_path,tmp_path,pd.Timestamp('2026-09-09T10:00Z'))[0]=='DUE'


def test_capture_lock_refuses_concurrent_or_stale_owner(tmp_path):
    with capture_lock(tmp_path):
        with pytest.raises(FileExistsError):
            with capture_lock(tmp_path): pass
    assert not (tmp_path/'.capture.lock').exists()


def test_capture_lock_released_after_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with capture_lock(tmp_path): raise RuntimeError('test interruption')
    assert not (tmp_path/'.capture.lock').exists()


def test_output_traversal_and_existing_path_rejected(tmp_path):
    build=tmp_path/'build'
    build.mkdir()
    assert fresh_output(tmp_path,build/'new') == build/'new'
    for candidate in [build,build/'..'/'outside']:
        with pytest.raises(ValueError): fresh_output(tmp_path,candidate)


def test_missing_query_dates_never_erase_saved_eex_history():
    history=pd.DataFrame(dict(date=pd.to_datetime(['2026-09-01','2026-09-04']),price=[1.,2.]))
    fresh=pd.DataFrame(dict(date=pd.to_datetime(['2026-09-04']),price=[3.]))
    result=merge_observed_eex_dates(history,fresh,['20260904'])
    assert result.price.tolist()==[1.,3.]


def test_eex_merge_rejects_unobserved_normalized_date():
    frame=pd.DataFrame(dict(date=pd.to_datetime(['2026-09-04']),price=[1.]))
    with pytest.raises(ValueError): merge_observed_eex_dates(frame,frame,['20260901'])


def test_quarantined_returned_date_never_erases_accepted_history():
    history = pd.DataFrame(dict(date=pd.to_datetime(['2026-09-04', '2026-09-07']), price=[1., 2.]))
    fresh = history.iloc[:1].copy()
    saved = history.copy(deep=True)
    with pytest.raises(ValueError, match='quarantin|accepted|surface'):
        merge_observed_eex_dates(history, fresh, ['20260904', '20260907'])
    pd.testing.assert_frame_equal(history, saved)


def test_empty_normalized_delta_cannot_replace_history():
    history = pd.DataFrame(dict(date=pd.to_datetime(['2026-09-07']), price=[2.]))
    with pytest.raises(ValueError, match='quarantin|accepted|empty'):
        merge_observed_eex_dates(history, history.iloc[:0], ['20260907'])


def test_unaccepted_newest_date_cannot_fall_back_silently():
    history = pd.DataFrame(dict(date=pd.to_datetime(['2026-09-04']), price=[2.]))
    with pytest.raises(ValueError, match='surface|accepted'):
        merge_observed_eex_dates(history, history, ['20260904', '20260907'])


def test_eex_merge_rejects_schema_drift():
    history = pd.DataFrame(dict(date=pd.to_datetime(['2026-09-04']), price=[2.]))
    fresh = history.rename(columns={'price': 'wrong_column'})
    with pytest.raises(ValueError, match='schema|columns'):
        merge_observed_eex_dates(history, fresh, ['20260904'])


def test_finish_records_actual_absence_of_stop_requests(tmp_path):
    queries = BoundedQueries(lambda *args: dict(state='RUNNING', auto_stop_mins=45), 'test', tmp_path)
    queries.finish()
    receipt = json.loads((tmp_path/'warehouse-final.json').read_text())
    assert receipt['stop_requests'] == 0
    assert receipt['manual_stop_attempted'] is False
    assert 'manual_stop' not in receipt


def capture_case(tmp_path, monkeypatch, *, fail_query=False, fail_finish=False):
    from types import SimpleNamespace

    from scripts import collect_lt_benchmark_day as collector

    build = tmp_path/'build'
    build.mkdir()
    out = build/'capture'
    out.mkdir()
    now = pd.Timestamp.now(tz='UTC')
    current = now.tz_convert('Europe/Zurich').tz_localize(None).normalize()
    for name, price in [('initial', 1.), ('registered', 2.)]:
        pd.DataFrame(dict(date=[current-pd.Timedelta(days=10)], price=[price])).to_parquet(build/(name+'.parquet'))
    (build/'recipe').write_bytes(b'frozen recipe')
    (build/'target').write_bytes(b'bound history')
    directory = build/'ompex'
    directory.mkdir()
    (directory/('HFC_Ompex_'+current.strftime('%Y%m%d')+'_101700.xlsx')).write_bytes(b'workbook fixture')
    def ref(name):
        return dict(path='build/'+name, sha256=collector.sha(build/name))
    config = dict(recipe=ref('recipe'), initial_eex_history=ref('initial.parquet'),
        ch_history=ref('target'), ompex_read_only_directory=str(directory), warehouse_id='test')
    records = [dict(entry=dict(recipe=config['recipe'], inputs={'EEX': {'artifact': ref('registered.parquet')}}))]
    monkeypatch.setattr(collector, 'parse_ompex_workbook', lambda *a, **kw:
        SimpleNamespace(data=pd.DataFrame({'price': [1.]}), receipt={}))
    fresh = pd.DataFrame(dict(date=[current-pd.Timedelta(days=1)], price=[3.]))
    monkeypatch.setattr(collector, 'normalize_databricks_eex_daily_snapshot', lambda *a, **kw:
        SimpleNamespace(history=fresh, quarantine=pd.DataFrame({'reason': []}), audit={'rows': 1}))
    class Queries:
        def __init__(self, *args):
            pass
        def ready(self):
            pass
        def query(self, label, *args, **kwargs):
            if fail_query:
                raise ValueError('primary capture failure')
            if label == 'EEX':
                raw = pd.DataFrame(dict(QuotationDateID=[fresh.date.iloc[0].strftime('%Y%m%d')],
                    FactLoadTimestampUtc=[now-pd.Timedelta(hours=1)]))
                raw.to_parquet(out/'sql/EEX.parquet')
                return raw
            if label == 'LSEG-dimension':
                return pd.DataFrame(dict(Country=['CHE'], Unit=['EUR/MWh'], ValueFrequency=['1h'], GroupName=['continuous_forward']))
            return pd.DataFrame({'price': [1.]})
        def finish(self):
            if fail_finish:
                raise RuntimeError('secondary final observation failure')
    monkeypatch.setattr(collector, 'BoundedQueries', Queries)
    monkeypatch.setattr(collector, 'normalize_lseg', lambda *args:
        (pd.Series([1.], name='price_eur_mwh', index=pd.date_range(current, periods=1, tz='UTC')), now.isoformat()))
    return collector, out, config, records


def test_day_two_capture_uses_first_registered_history(tmp_path, monkeypatch):
    collector, out, config, records = capture_case(tmp_path, monkeypatch)
    request = collector.capture(tmp_path, out, config, records, request=None)
    assert pd.read_parquet(out/'EEX.parquet').price.tolist() == [2., 3.]
    assert len(set(request['inputs']['EEX']['quotation_dates'].values())) == 1
    receipt = json.loads((out/'eex-history-merge.json').read_text())
    assert receipt['previous_history'] == records[0]['entry']['inputs']['EEX']['artifact']


def test_final_warehouse_failure_preserves_primary_capture_error(tmp_path, monkeypatch):
    collector, out, config, records = capture_case(tmp_path, monkeypatch, fail_query=True, fail_finish=True)
    with pytest.raises(ValueError, match='primary capture failure'):
        collector.capture(tmp_path, out, config, records, request=None)
    receipt = json.loads((out/'sql/warehouse-final-failure.json').read_text())
    assert receipt['type'] == 'RuntimeError'
    assert receipt['stop_requests'] == 0


@pytest.mark.parametrize('failure', ['registry', 'config'])
def test_early_failure_after_safe_output_has_durable_receipt(tmp_path, monkeypatch, failure):
    import sys

    from scripts import collect_lt_benchmark_day as collector

    monkeypatch.setattr(collector, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    build = tmp_path/'build'
    build.mkdir()
    config = build/'config.json'
    config.write_text('{}')
    registry = build/'registry'
    registry.mkdir()
    out = build/'attempt'
    if failure == 'registry':
        (registry/'0001-invalid.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['collector', '--config', str(config), '--registry', str(registry), '--output', str(out)])
    with pytest.raises(ValueError):
        collector.main()
    receipt = json.loads((out/'failure.json').read_text())
    assert receipt['type'] == 'ValueError'
    assert 'failure.json' in json.loads((out/'manifest.json').read_text())
