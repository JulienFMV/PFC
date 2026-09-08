"""One bounded local EEX/OMPEX/LSEG capture and existing D304 daily builder.

Callable by a governed scheduler; never installs one or registers a fake day.
Identical-day invocations exit before credentials, SQL, or Warehouse access.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.databricks_eex_daily_snapshot import normalize_databricks_eex_daily_snapshot
from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.path_safety import assert_absolute_path_has_no_links
from pfc_shaping.validation.lt_benchmark_snapshots import bound_file, read_registry, utc
from pfc_shaping.validation.ompex_benchmark import parse_ompex_workbook

ROOT = Path(r'C:\Users\jbattaglia\PFC_LT')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def due_day(root, registry, now):
    records, _ = read_registry(root, registry, now=now)
    if len(records) >= 20:
        return 'PILOT_COMPLETE', records
    if records and utc(records[-1]['entry']['valuation_at_utc']).tz_convert('Europe/Zurich').date() == utc(now).tz_convert('Europe/Zurich').date():
        return 'ALREADY_CAPTURED_TODAY', records
    return 'DUE', records


def fresh_output(root, path):
    raw = Path(path).absolute()
    resolved = raw.resolve()
    if not resolved.is_relative_to(root/'build') or resolved == root/'build' or resolved.exists():
        raise ValueError('fresh build output required')
    assert_absolute_path_has_no_links(raw)
    return resolved


@contextmanager
def capture_lock(registry):
    """An interrupted owner's lock is an explicit recovery condition, never stolen."""
    path = registry/'.capture.lock'
    assert_absolute_path_has_no_links(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(pd.Timestamp.now(tz='UTC').isoformat())
        stream.flush()
    try:
        yield
    finally:
        path.unlink()


class BoundedQueries:
    """Exact finite read budget; HTTP callable injected for deterministic tests."""
    def __init__(self, request, warehouse, out, *, monotonic=time.monotonic, sleep=time.sleep):
        self.request, self.warehouse, self.out = request, warehouse, out
        self.monotonic, self.sleep = monotonic, sleep
        self.count, self.started = 0, False

    def ready(self):
        state = self.request('GET', '/api/2.0/sql/warehouses/'+self.warehouse)
        write(self.out/'warehouse-before.json', {k:state.get(k) for k in ['id','state','auto_stop_mins','cluster_size']})
        if not isinstance(state.get('auto_stop_mins'), int) or not 0 < state['auto_stop_mins'] <= 45:
            raise ValueError('bounded configured auto-stop required; no configuration change')
        if state['state'] == 'STOPPED':
            self.request('POST', '/api/2.0/sql/warehouses/'+self.warehouse+'/start')
            self.started = True
        elif state['state'] != 'RUNNING':
            raise ValueError('existing Warehouse transition; no interference')
        deadline = self.monotonic()+600
        while state['state'] != 'RUNNING':
            if self.monotonic() >= deadline:
                raise TimeoutError('Warehouse start exceeded 600 seconds')
            self.sleep(3)
            state = self.request('GET', '/api/2.0/sql/warehouses/'+self.warehouse)

    def query(self, label, sql, *, row_limit):
        if self.count >= 6 or not 0 < row_limit <= 30000 or not sql.lstrip().upper().startswith('SELECT '):
            raise ValueError('bounded SELECT budget exceeded')
        self.count += 1
        (self.out/(label+'.sql')).write_text(sql+'\n', encoding='utf-8')
        deadline = self.monotonic()+180
        result = self.request('POST', '/api/2.0/sql/statements', dict(warehouse_id=self.warehouse, statement=sql,
            disposition='INLINE', format='JSON_ARRAY', wait_timeout='10s', on_wait_timeout='CONTINUE', row_limit=row_limit))
        sid = result['statement_id']
        try:
            while result['status']['state'] in ('PENDING','RUNNING'):
                if self.monotonic() >= deadline:
                    raise TimeoutError('statement exceeded 180 seconds')
                self.sleep(3)
                result = self.request('GET', '/api/2.0/sql/statements/'+sid)
            write(self.out/(label+'-response.json'), result)
            if result['status']['state'] != 'SUCCEEDED' or result['manifest'].get('truncated') is not False:
                raise ValueError('failed or truncated statement')
            chunk = result.get('result', {})
            rows = list(chunk.get('data_array', []))
            visited = set()
            while chunk.get('next_chunk_internal_link'):
                link = chunk['next_chunk_internal_link']
                if not link.startswith('/api/2.0/sql/statements/'+sid+'/result/chunks/') or link in visited:
                    raise ValueError('invalid result pagination')
                if self.monotonic() >= deadline or len(visited) >= 20:
                    raise TimeoutError('result pagination deadline/cap')
                visited.add(link)
                chunk = self.request('GET', link)
                rows.extend(chunk.get('data_array', []))
                if len(rows) >= row_limit:
                    raise ValueError('result row limit reached')
            if len(rows) >= row_limit or len(rows) != result['manifest']['total_row_count']:
                raise ValueError('incomplete result or rejection sentinel reached')
            if self.monotonic() >= deadline:
                raise TimeoutError('completed result exceeded 180 seconds')
            frame = pd.DataFrame(rows, columns=[c['name'] for c in result['manifest']['schema']['columns']])
            frame.to_parquet(self.out/(label+'.parquet'), index=False)
            return frame
        except BaseException:
            if result['status']['state'] in ('PENDING','RUNNING'):
                self.request('POST', '/api/2.0/sql/statements/'+sid+'/cancel')
            raise

    def finish(self):
        state = self.request('GET', '/api/2.0/sql/warehouses/'+self.warehouse)
        write(self.out/'warehouse-final.json', dict(state=state.get('state'), auto_stop_mins=state.get('auto_stop_mins'),
            warehouse_started=self.started, statements=self.count, table_writes=0,
            manual_stop='PRIOR_HTTP403_NOT_RETRIED', shutdown_confirmed=state.get('state') == 'STOPPED'))


def normalize_lseg(frame, observed):
    if frame.empty or frame.Unit.ne('EUR/MWh').any():
        raise ValueError('nonempty EUR/MWh LSEG source required')
    index = pd.DatetimeIndex(pd.to_datetime(frame.ValueStartDateTimeUtc, utc=True))
    issue = pd.to_datetime(frame.ForecastIssuedAtUtc, utc=True)
    if issue.isna().any() or issue.nunique() != 1 or (issue > observed).any():
        raise ValueError('one pre-observation LSEG vintage required')
    for field in ['PipelineFirstSeenAtUtc','PullTimestampUtc']:
        stamps = pd.to_datetime(frame[field], utc=True)
        if stamps.isna().any() or (stamps > observed).any() or (stamps < issue).any():
            raise ValueError('LSEG availability chronology failed')
    ends = pd.DatetimeIndex(pd.to_datetime(frame.DateTimeUtc, utc=True))
    values = pd.to_numeric(frame.Value, errors='raise').to_numpy(dtype=float)
    if (not index.is_unique or not index.is_monotonic_increasing or not np.isfinite(values).all()
            or not (ends-index == pd.Timedelta(hours=1)).all()
            or (len(index)>1 and not (index[1:]-index[:-1] == pd.Timedelta(hours=1)).all())):
        raise ValueError('unique contiguous native LSEG hourly intervals required')
    return pd.Series(values, index=index, name='price_eur_mwh'), issue.iloc[0].isoformat()


def merge_observed_eex_dates(history, normalized, quotation_dates):
    """Only dates actually returned by the source replace saved history."""
    dates = pd.to_datetime(pd.Series(quotation_dates).astype(str)).dt.normalize()
    if dates.empty or dates.isna().any() or not pd.to_datetime(normalized.date).isin(dates).all():
        raise ValueError('normalized rows must belong to observed quotation dates')
    return pd.concat([history.loc[~pd.to_datetime(history.date).isin(dates)], normalized], ignore_index=True)


def capture(root, out, config, records, *, request):
    now = pd.Timestamp.now(tz='UTC')
    day = now.tz_convert('Europe/Zurich').strftime('%Y%m%d')
    recipe = bound_file(root, config['recipe'])
    if records and config['recipe'] != records[0]['entry']['recipe']:
        raise ValueError('frozen recipe changed')
    history_ref = records[-1]['entry']['inputs']['EEX']['artifact'] if len(records)>1 else config['initial_eex_history']
    history = pd.read_parquet(bound_file(root, history_ref))
    if 'date' not in history:
        raise ValueError('normalized historical EEX input required')
    target = bound_file(root, config['ch_history'])
    directory = Path(config['ompex_read_only_directory'])
    matches = sorted(directory.glob('HFC_Ompex_'+day+'_*.xlsx'))
    if len(matches) != 1:
        raise ValueError('exactly one current-day OMPEX workbook required; no stale fallback')
    source = matches[0]
    before = source.stat()
    payload = source.read_bytes()
    after = source.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns) or len(payload) != before.st_size:
        raise ValueError('OMPEX changed during capture')
    ompex_seen = pd.Timestamp.now(tz='UTC')
    (out/source.name).write_bytes(payload)
    parsed = parse_ompex_workbook(payload, source_name=source.name, accept_unconfirmed_hour_end_semantics=True)
    parsed.data.to_parquet(out/'OMPEX.parquet', index=False)
    write(out/'ompex-receipt.json', dict(source_name=source.name, observed_at_utc=ompex_seen.isoformat(),
        sha256=sha(out/source.name), structure=dict(parsed.receipt), vendor_availability_authenticated=False))
    sql_dir = out/'sql'
    sql_dir.mkdir()
    queries = BoundedQueries(request, config['warehouse_id'], sql_dir)
    try:
        queries.ready()
        start = (now.tz_convert('Europe/Zurich').normalize()-pd.Timedelta(days=7)).strftime('%Y%m%d')
        raw = queries.query('EEX', f"SELECT f.ProductID, f.DeliveryPeriodID, f.QuotationDateID, CAST(f.SettlementPrice AS DOUBLE) AS SettlementPriceEurMWh, CAST(f.LastPrice AS DOUBLE) AS LastPriceEurMWh, f.Meta_Load_Timestamp AS FactLoadTimestampUtc, p.Country, p.Commodity, p.ProductType, p.DeliveryPeriodType, d.DeliveryStartDate, d.DeliveryEndDate FROM prd.gold.facteexpricedaily f JOIN prd.gold.dimeexproduct p ON p.ProductID=f.ProductID JOIN prd.gold.dimeexdeliveryperiod d ON d.DeliveryPeriodID=f.DeliveryPeriodID WHERE p.Country='CH' AND p.Commodity='POWER' AND f.QuotationDateID>={start} AND f.QuotationDateID<={day} ORDER BY f.QuotationDateID, f.ProductID, f.DeliveryPeriodID", row_limit=5000)
        eex_seen = pd.Timestamp.now(tz='UTC')
        if (pd.to_datetime(raw.FactLoadTimestampUtc, utc=True) > eex_seen).any():
            raise ValueError('EEX loaded after observation')
        normalized = normalize_databricks_eex_daily_snapshot(raw, source_snapshot_sha256=sha(sql_dir/'EEX.parquet'), as_of_date=now.tz_convert('Europe/Zurich').date())
        merged = merge_observed_eex_dates(history, normalized.history, raw.QuotationDateID)
        merged.to_parquet(out/'EEX.parquet', index=False)
        write(out/'eex-normalization.json', dict(normalized.audit))
        normalized.quarantine.to_parquet(out/'eex-quarantine.parquet', index=False)
        dimension = queries.query('LSEG-dimension', "SELECT CurveID, GroupName, Country, Unit, ValueFrequency, VendorTimezone FROM prd.gold.dimlsegcurves WHERE CurveID = '110181967'", row_limit=3)
        if len(dimension)!=1 or dimension.Country.iloc[0]!='CHE' or dimension.Unit.iloc[0]!='EUR/MWh' or dimension.ValueFrequency.iloc[0]!='1h' or dimension.GroupName.iloc[0]!='continuous_forward':
            raise ValueError('frozen LSEG curve semantics changed')
        frames = []
        for year in [2026,2027,2028,2029]:
            begin = '2026-09-30 22:00:00' if year==2026 else f'{year-1}-12-31 23:00:00'
            end = f'{year}-12-31 23:00:00'
            frames.append(queries.query('LSEG-'+str(year), f"SELECT ValueStartDateTimeUtc, DateTimeUtc, Value, Unit, ForecastIssuedAtUtc, PipelineFirstSeenAtUtc, PullTimestampUtc FROM prd.gold.factlsegcurvevalueslatest WHERE CurveID='110181967' AND ScenarioID=0 AND ValueStartDateTimeUtc>=TIMESTAMP '{begin}' AND ValueStartDateTimeUtc<TIMESTAMP '{end}' ORDER BY ValueStartDateTimeUtc", row_limit=10000))
        lseg_seen = pd.Timestamp.now(tz='UTC')
        series, issue = normalize_lseg(pd.concat(frames, ignore_index=True), lseg_seen)
        series.to_frame().to_parquet(out/'LSEG.parquet')
    finally:
        queries.finish()
    def ref(path):
        return dict(path=path.relative_to(root).as_posix(), sha256=sha(path))
    inputs = {}
    for role, path, seen, issued in [('EEX',out/'EEX.parquet',eex_seen,None), ('CH_HISTORY',target,now,None),
            ('OMPEX',out/'OMPEX.parquet',ompex_seen,None), ('LSEG',out/'LSEG.parquet',lseg_seen,issue)]:
        inputs[role] = dict(artifact=ref(path), observed_at_utc=seen.isoformat(), issue_at_utc=issued,
            vendor_availability_authenticated=False)
    bound_file(root, config['recipe'])
    bound_file(root, config['ch_history'])
    return dict(recipe=ref(recipe), inputs=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        raise ValueError('canonical workspace required')
    output = fresh_output(ROOT, args.output)
    status, records = due_day(ROOT, args.registry, pd.Timestamp.now(tz='UTC'))
    output.mkdir(parents=True, exist_ok=False)
    if status != 'DUE':
        write(output/'status.json', dict(status=status, network_calls=0, authority=dict(AUTHORITIES)))
        print(json.dumps(dict(status=status, network_calls=0)))
        return
    config_path = args.config.resolve(strict=True)
    if not config_path.is_relative_to(ROOT/'build'):
        raise ValueError('local capture config required')
    config = json.loads(config_path.read_text())
    if set(config) != {'recipe','initial_eex_history','ch_history','ompex_read_only_directory','warehouse_id'}:
        raise ValueError('exact collector config schema required')
    # Preserve the reviewed configuration and code before any external call.
    write(output/'plan.json', dict(config=config, config_sha256=sha(config_path), collector_sha256=sha(Path(__file__)),
        started_at_utc=pd.Timestamp.now(tz='UTC').isoformat(), authority=dict(AUTHORITIES)))
    from scripts.capture_entsoe_day_ahead_prd_profile import _load_credentials, _request_json
    host, warehouse, token = _load_credentials()
    if warehouse != config['warehouse_id']:
        raise ValueError('configured Warehouse differs from credential boundary')
    def request(method, path, body=None):
        return _request_json(method=method, url=host+path, token=token, body=body, timeout_seconds=30)
    try:
        with capture_lock(args.registry.resolve(strict=True)):
            status, records = due_day(ROOT, args.registry, pd.Timestamp.now(tz='UTC'))
            if status != 'DUE':
                write(output/'status.json', dict(status=status, network_calls=0, authority=dict(AUTHORITIES)))
                return
            request_data = capture(ROOT, output, config, records, request=request)
            write(output/'request.json', request_data)
            subprocess.run([sys.executable, '-B', '-m', 'scripts.run_lt_benchmark_day', '--request', str(output/'request.json'),
                '--registry', str(args.registry), '--output', str(output/'snapshot')], cwd=ROOT, check=True)
            write(output/'status.json', dict(status='LOCAL_CAPTURE_AND_D304_DAY_COMPLETE', authority=dict(AUTHORITIES)))
    except Exception as exc:
        write(output/'failure.json', dict(type=type(exc).__name__, message=str(exc), authority=dict(AUTHORITIES)))
        raise
    finally:
        write(output/'manifest.json', {p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})


if __name__ == '__main__':
    main()
