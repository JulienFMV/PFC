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
