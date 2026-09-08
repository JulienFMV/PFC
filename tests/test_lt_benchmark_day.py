import copy
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from pfc_shaping.lt.local_benchmark import AUTHORITIES
from pfc_shaping.validation.lt_benchmark_snapshots import score_closed_months
from scripts.run_lt_benchmark_day import preflight


def request_fixture(tmp_path):
    build = tmp_path/'build'
    build.mkdir()
    registry = build/'registry'
    registry.mkdir()
    (build/'data').write_bytes(b'fixture')
    recipe = dict(model='signed-equal', authority=dict(AUTHORITIES), external_model_input=False, source_code_sha256={})
    (build/'recipe.json').write_text(json.dumps(recipe))
    def ref(name):
        return dict(path='build/'+name,sha256=hashlib.sha256((build/name).read_bytes()).hexdigest())
    request = dict(recipe=ref('recipe.json'), inputs={role:dict(artifact=ref('data'),observed_at_utc='2026-09-08T09:00Z',
        issue_at_utc=None,vendor_availability_authenticated=False) for role in ['EEX','CH_HISTORY','OMPEX','LSEG']})
    return registry,request


def test_preflight_rejects_next_day_stale_capture(tmp_path):
    registry,request=request_fixture(tmp_path)
    with pytest.raises(ValueError,match='re-observed'):
        preflight(tmp_path,request,registry,'2026-09-09T10:00Z')


def test_preflight_accepts_frozen_local_inputs(tmp_path):
    registry,request=request_fixture(tmp_path)
    recipe,paths=preflight(tmp_path,request,registry,'2026-09-08T10:00Z')
    assert recipe['model']=='signed-equal' and set(paths)==set(request['inputs'])


def test_preflight_rejects_external_as_model_input(tmp_path):
    registry,request=request_fixture(tmp_path)
    path=tmp_path/'build/recipe.json'
    recipe=json.loads(path.read_text())
    recipe['external_model_input']=True
    path.write_text(json.dumps(recipe))
    request['recipe']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='D304'):
        preflight(tmp_path,request,registry,'2026-09-08T10:00Z')


def test_level_shift_has_zero_shape_error():
    index=pd.date_range('2026-10-01','2026-12-01',freq='h',inclusive='left',tz='Europe/Zurich').tz_convert('UTC')
    truth=pd.Series(20+np.sin(np.arange(len(index))),index=index)
    result=score_closed_months(truth+5,truth,truth_available_at='2026-12-02T00:00Z',now='2026-12-03T00:00Z',committed_at='2026-09-08T12:00Z')
    assert len(result['months'])==2
    for month in result['months']:
        metrics=month['diagnostics']
        assert float(metrics['monthly_mean_error_eur_mwh'])==pytest.approx(5)
        assert float(metrics['zero_mean_hourly_shape']['mae_eur_mwh'])<1e-12


def test_late_commitment_and_missing_truth_rejected():
    index=pd.date_range('2026-10-01','2026-11-01',freq='h',inclusive='left',tz='Europe/Zurich').tz_convert('UTC')
    truth=pd.Series(np.ones(len(index)),index=index)
    with pytest.raises(ValueError,match='late prediction'):
        score_closed_months(truth,truth,truth_available_at='2026-11-02T00:00Z',now='2026-11-03T00:00Z',committed_at='2026-10-02T12:00Z')
    with pytest.raises(ValueError,match='complete Swiss'):
        score_closed_months(truth,truth.iloc[:-1],truth_available_at='2026-11-02T00:00Z',now='2026-11-03T00:00Z',committed_at='2026-09-08T12:00Z')
