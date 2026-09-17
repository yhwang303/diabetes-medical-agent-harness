"""P04 controlled RL MCP: real SDK/worker integration and explicit fault probes, all offline."""
from dataclasses import replace
import json
import sqlite3
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import httpx
import pytest

from conftest import allow_data
from medical_harness.api import create_app
from medical_harness.contracts import GateError, canonical, digest
from medical_harness.core import Core
from medical_harness.paths import ROOT
from medical_harness.rl_agent import RLAgent
from medical_harness.sdk_worker import RL_OPERATIONS, PREDICTION_OPERATIONS, TOOL_PREFIX, permitted
from medical_harness.store import Store
from test_prediction_agent import setup as prediction_setup, counts, latest_result
from test_report_agent import FaultRunner, completed
from test_sdk import SSE, message_events


def setup(core, run, tmp_path, *, action=None, provider=None, authorize=True):
    if authorize: allow_data(core, run, 'rl')
    key = tmp_path / 'offline-rl.key'; key.write_text('offline-rl-secret'); key.chmod(0o600)
    return RLAgent(core, key, tmp_path / 'rl-agent', runner=FaultRunner(action) if action else None,
        transport=httpx.MockTransport(provider or (lambda req: httpx.Response(200,
            stream=SSE(message_events({'type': 'text', 'text': 'done'}))))))


def controlled_prediction(core, run, tmp_path):
    def action(client, kwargs):
        assert client.post('/tools/request_prediction', json={}).json()['prediction_available']
        return completed()
    agent = prediction_setup(core, run, tmp_path, action=action)
    try: agent.run('alice', run['id'])
    finally: agent.close()


def success(client, kwargs):
    result = client.post('/tools/request_rl', json={}).json()
    assert result['policy_available'], result
    assert client.post('/tools/inspect_rl', json={'job_id': result['job_id']}).json() == result
    return completed()


def provider_for(operations, requests, references, *, repeat=True):
    request_name, query_name = operations
    available = 'prediction_available' if operations == PREDICTION_OPERATIONS else 'policy_available'
    def provider(req):
        body = json.loads(req.content); requests.append(body)
        assert {t['name'] for t in body['tools']} == {TOOL_PREFIX + name for name in operations}
        phase = len(requests)
        if phase == 1: name, args = request_name, {}
        else:
            previous = latest_result(body); references.append(previous)
            assert previous[available] and previous['evidence']['origin'] == 'fixture'
            if phase == 2 and repeat: name, args = request_name, {}
            elif phase == (3 if repeat else 2): name, args = query_name, {'job_id': previous['job_id']}
            else: return httpx.Response(200, stream=SSE(message_events({'type': 'text', 'text': 'No clinical advice.'})))
        return httpx.Response(200, stream=SSE(message_events({'type': 'tool_use', 'id': 'rl-test-' + str(phase),
            'name': TOOL_PREFIX + name, 'input': args})))
    return provider


@pytest.mark.parametrize('basal', [False, True])
def test_actual_two_sdk_sessions_and_independent_workers_bind_same_parent(tmp_path, snapshot, basal):
    core = Core(Store(tmp_path / 'real-sdk.sqlite3'), enable_fixtures=True, enable_basal_fixtures=basal)
    if basal: snapshot = json.loads((ROOT / 'examples/basal-research-case.json').read_text())
    case = core.create_case('alice', snapshot)
    run = core.start_run('alice', case['case_id'], 'eval', 'two-sdk')
    pr, pref, rr, rref = [], [], [], []
    pred = prediction_setup(core, run, tmp_path, provider=provider_for(PREDICTION_OPERATIONS, pr, pref, repeat=False))
    rl = setup(core, run, tmp_path, provider=provider_for(RL_OPERATIONS, rr, rref))
    try:
        prediction = pred.run('alice', run['id'])
        candidate = rl.run('alice', run['id'])
        assert rref[0] == rref[1] == rref[2] and len(rr) == 4
        assert candidate == rl.run('alice', run['id']) and len(rr) == 4
        assert set(candidate) == {'ok','job_id','status','policy_available','clinical_use','safety_checked','evidence','sdk_job_id'}
        assert not candidate['safety_checked'] and not candidate['clinical_use']
        assert core.status('alice', run['id'])['state'] == 'POLICY_ACCEPTED'
        assert counts(core) == {'jobs':4,'artifacts':2,'drafts':0,'reviews':0,'releases':0}
        with core.store.tx() as db:
            policy = json.loads(db.execute("SELECT body FROM artifacts WHERE kind='policy'").fetchone()[0])
            assert policy['parent_hash'] == policy['payload']['forecast_parent_hash'] == prediction['evidence']['artifact_hash']
            starts = [json.loads(r[0]) for r in db.execute("SELECT body FROM events WHERE kind='worker_started'")]
            assert len({r['pid'] for r in starts}) == 4
            assert db.execute('SELECT count(*) FROM job_dependencies').fetchone()[0] == 2
            assert db.execute('SELECT count(*) FROM job_data_permissions').fetchone()[0] == 4
            assert db.execute("SELECT count(*) FROM sdk_sessions WHERE state!='REMOVED'").fetchone()[0] == 0
        for forbidden in ('source_ref', 'missing_mask', 'action_value', 'action_unit', 'offline-rl-secret', 'delivered_units'):
            assert forbidden not in canonical(rr)
        assert '"values"' not in canonical(rref)
        # Separate engineering safety gate still works once both SDK sessions completed.
        assert core.execute_models('alice', run['id'])['state'] == 'SAFETY_ACCEPTED'
        assert core.input_timeline('alice', case['case_id'])['snapshot'] == snapshot
        (tmp_path / 'rl-sdk-proof.json').write_text(canonical({'basal':basal,'counts':counts(core),
            'prediction_reference':prediction['evidence'],'candidate_reference':candidate['evidence'],
            'worker_pids':[r['pid'] for r in starts],'paid_calls':0,'online_provider':False}))
    finally: pred.close(); rl.close(); core.close()


@pytest.mark.parametrize('precondition', ['no_permission','report_permission','prediction_permission','missing_prediction','failed_prediction','expired','wrong_owner','missing_key'])
def test_preconditions_before_sdk_and_key_access(core, run, tmp_path, precondition):
    if precondition not in ('missing_prediction','failed_prediction'): core.execute_prediction('alice', run['id'])
    if precondition == 'report_permission': allow_data(core, run, 'report')
    elif precondition == 'prediction_permission': allow_data(core, run, 'prediction')
    elif precondition not in ('no_permission',): allow_data(core, run, 'rl')
    if precondition == 'failed_prediction':
        with core.store.tx() as db: db.execute("UPDATE runs SET state='UNAVAILABLE',reason='EXECUTOR_TIMEOUT'")
    if precondition == 'expired':
        with core.store.tx() as db: deadline = db.execute('SELECT expires FROM runs WHERE id=?',(run['id'],)).fetchone()[0]
        core.clock = lambda: deadline + 1
    agent = RLAgent(core, tmp_path / 'missing.key', tmp_path / 'never-sdk')
    try:
        before = counts(core)
        with pytest.raises(GateError): agent.run('bob' if precondition == 'wrong_owner' else 'alice', run['id'])
        assert counts(core) == before and not agent.evidence
        if precondition != 'missing_key': assert not agent.data_dir.exists()
    finally: agent.close()


def test_prediction_sdk_must_complete_before_rl_or_ordinary_downstream(core, run, tmp_path):
    allow_data(core, run, 'rl')
    rl = RLAgent(core, tmp_path / 'unread.key', tmp_path / 'uncreated')
    def action(client, kwargs):
        assert client.post('/tools/request_prediction', json={}).json()['prediction_available']
        with pytest.raises(GateError, match='PREDICTION_AGENT_INCOMPLETE'): rl.run('alice', run['id'])
        with pytest.raises(GateError, match='PREDICTION_AGENT_INCOMPLETE'): core._model_step('alice', run['id'], 'policy')
        assert counts(core)['jobs'] == 2 and not rl.data_dir.exists()
        return completed()
    pred = prediction_setup(core, run, tmp_path, action=action)
    try: assert pred.run('alice', run['id'])['prediction_available']
    finally: pred.close(); rl.close()


@pytest.mark.parametrize('payload', [{'prediction':{}},{'forecast_parent_hash':'a'*64},{'case_id':'a'*32},
                                    {'action_value':0},{'origin':'model'},{'producer':'trusted'},{'run_id':'a'*32}])
def test_no_client_inputs_or_result_registration(core, run, tmp_path, payload):
    core.execute_prediction('alice', run['id'])
    def action(client, kwargs):
        assert client.post('/tools/request_rl', json=payload).json()['error'] == 'INVALID_TOOL_ARGUMENTS'
        assert counts(core)['artifacts'] == 1
        return success(client, kwargs)
    agent = setup(core, run, tmp_path, action=action)
    try: assert agent.run('alice', run['id'])['policy_available']
    finally: agent.close()


def test_cross_case_query_and_ordinary_api_cannot_bypass_rl_session(core, run, snapshot, tmp_path):
    core.execute_prediction('alice', run['id'])
    other = core.create_case('alice', snapshot)
    other_run = core.start_run('alice', other['case_id'], 'eval', 'other'); core.execute_models('alice', other_run['id'])
    with core.store.tx() as db:
        other_job = db.execute("SELECT id FROM jobs WHERE run_id=? AND step='policy'",(other_run['id'],)).fetchone()[0]
    def action(client, kwargs):
        with pytest.raises(GateError, match='RL_SESSION_REQUIRED'): core.execute_models('alice', run['id'])
        assert client.post('/tools/inspect_rl', json={'job_id':other_job}).json()['error'] == 'NOT_FOUND'
        assert client.post('/tools/inspect_rl', json={'job_id':'../private'}).json()['error'] == 'INVALID_TOOL_ARGUMENTS'
        return success(client, kwargs)
    agent = setup(core, run, tmp_path, action=action)
    try: assert agent.run('alice', run['id'])['policy_available']
    finally: agent.close()


@pytest.mark.parametrize('scenario,code', [('abstain','POLICY_ABSTAIN'),('unsupported','POLICY_UNSUPPORTED'),
    ('error','POLICY_ERROR'),('invalid_output','INVALID_MODEL_OUTPUT'),('parent_mismatch','FORECAST_PARENT_MISMATCH')])
def test_non_candidate_outputs_never_succeed_or_retry(core, snapshot, tmp_path, scenario, code):
    case = core.create_case('alice', snapshot)
    run = core.start_run('alice', case['case_id'], 'eval', 'failure', fixture_scenario=scenario)
    core.execute_prediction('alice', run['id'])
    def action(client, kwargs):
        result = client.post('/tools/request_rl', json={}).json()
        assert not result['policy_available'] and result['error'] == code
        assert not client.post('/tools/request_rl', json={}).json()['policy_available']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError): agent.run('alice', run['id'])
        assert counts(core)['artifacts'] == 1
        with core.store.tx() as db: assert db.execute("SELECT count(*) FROM jobs WHERE step='policy'").fetchone()[0] == 1
        with pytest.raises(GateError): agent.run('alice', run['id'])
    finally: agent.close()


@pytest.mark.parametrize('fault', ['parent_permission','child_permission','dependency','parent_job','prediction_hash','rl_config','snapshot','withdraw','withdraw_regrant','delete','cancel'])
def test_inflight_rl_revalidates_all_dependencies(core, run, snapshot, tmp_path, fault, monkeypatch):
    controlled_prediction(core, run, tmp_path)
    original = core._fixtures['policy']
    def corrupt(request):
        if fault == 'snapshot': core.update_case('alice', run['case_id'], snapshot)
        elif fault == 'cancel': core.cancel('alice', run['id'])
        elif fault == 'delete':
            core.data_lifecycle.delete('alice', run['case_id'], {'snapshot_id':run['snapshot_id'], 'contract_version':'engineering-lifecycle-v1'})
        elif fault in ('withdraw','withdraw_regrant'):
            core.data_permissions.update('alice', run['case_id'], {'snapshot_id':run['snapshot_id'],'purpose':'rl','allowed':False,'policy_version':'engineering-data-policy-v1'})
            if fault == 'withdraw_regrant': allow_data(core, run, 'rl')
        elif fault == 'rl_config':
            import medical_harness.rl_agent as module
            old = module.rl_configuration()
            monkeypatch.setattr(module,'rl_configuration',lambda: dict(old,version='changed'))
        else:
            with core.store.tx() as db:
                if fault.endswith('permission'):
                    step = 'rl_agent' if fault == 'parent_permission' else 'policy'
                    db.execute('DELETE FROM job_data_permissions WHERE job_id IN (SELECT id FROM jobs WHERE step=?)',(step,))
                elif fault == 'dependency': db.execute("DELETE FROM job_dependencies WHERE child_job_id IN (SELECT id FROM jobs WHERE step='policy')")
                elif fault == 'parent_job': db.execute("UPDATE jobs SET status='FAILED' WHERE step='rl_agent'")
                else: db.execute("UPDATE artifacts SET digest=? WHERE kind='prediction'",('a'*64,))
        return original.call(request)
    core._fixtures['policy'] = replace(original,call=corrupt)
    def action(client,kwargs):
        assert not client.post('/tools/request_rl',json={}).json()['policy_available']
        return completed()
    agent = setup(core, run, tmp_path, action=action)
    try:
        with pytest.raises(GateError): agent.run('alice', run['id'])
        with core.store.tx() as db: assert not db.execute("SELECT 1 FROM artifacts WHERE kind='policy'").fetchone()
    finally: agent.close()


@pytest.mark.parametrize('fault',['prediction_permission','rl_permission','prediction_dependency','rl_dependency','prediction_config','rl_config','prediction_parent','rl_parent'])
def test_completed_candidates_revalidate_sdk_authority_before_downstream(core, run, tmp_path, monkeypatch, fault):
    controlled_prediction(core, run, tmp_path)
    agent = setup(core, run, tmp_path, action=success)
    try:
        assert agent.run('alice',run['id'])['policy_available']
        if fault.endswith('config'):
            module = __import__('medical_harness.'+('prediction_agent' if fault.startswith('prediction') else 'rl_agent'),fromlist=['x'])
            function = 'prediction_configuration' if fault.startswith('prediction') else 'rl_configuration'
            original = getattr(module,function)()
            monkeypatch.setattr(module,function,lambda:dict(original,version='changed'))
        else:
            kind = 'prediction' if fault.startswith('prediction') else 'policy'
            with core.store.tx() as db:
                if fault.endswith('permission'):
                    db.execute('DELETE FROM job_data_permissions WHERE job_id IN (SELECT id FROM jobs WHERE step=?)',(kind,))
                elif fault.endswith('dependency'):
                    db.execute('DELETE FROM job_dependencies WHERE child_job_id IN (SELECT id FROM jobs WHERE step=?)',(kind,))
                else: db.execute("UPDATE jobs SET status='FAILED' WHERE step=?",('prediction_agent' if kind=='prediction' else 'rl_agent',))
        with pytest.raises(GateError): agent.run('alice',run['id'])
        with pytest.raises(GateError): core.execute_models('alice',run['id'])
        with pytest.raises(GateError): core.create_draft_job('alice',run['id'])
        assert counts(core)['releases'] == 0
    finally: agent.close()


def test_rl_sdk_failure_after_candidate_cannot_enable_safety_or_draft(core,run,tmp_path):
    core.execute_prediction('alice',run['id'])
    def action(client,kwargs):
        result = client.post('/tools/request_rl',json={}).json()
        assert result['policy_available']
        with core.store.tx() as db:
            with pytest.raises(GateError,match='RL_AGENT_INCOMPLETE'): core._evidence(db,core._run(db,run['id'],'alice'))
        return {'results':[{'is_error':True,'terminal_reason':'error'}]}
    agent = setup(core,run,tmp_path,action=action)
    try:
        with pytest.raises(GateError,match='RL_AGENT_INCOMPLETE'):agent.run('alice',run['id'])
        assert not core.status('alice',run['id'])['currently_valid']
        with pytest.raises(GateError):core.create_draft_job('alice',run['id'])
    finally:agent.close()


def test_rl_profile_does_not_expand_any_existing_role():
    assert permitted(TOOL_PREFIX+'request_rl',{},profile='rl')
    for name in ('Bash','Read','Agent',TOOL_PREFIX+'request_prediction',TOOL_PREFIX+'submit_review',
                 TOOL_PREFIX+'submit_report_proposal',TOOL_PREFIX+'release_probe',TOOL_PREFIX+'register_policy'):
        assert not permitted(name,{},profile='rl')
    assert not permitted(TOOL_PREFIX+'request_rl',{},profile='rl',child=True)
    for profile in ('prediction','report','review','smoke'):
        assert not permitted(TOOL_PREFIX+'request_rl',{},profile=profile)


def test_http_default_disabled_auth_body_and_idempotence(core,run,tmp_path):
    core.execute_prediction('alice',run['id'])
    agent = setup(core,run,tmp_path,action=success)
    path='/runs/'+run['id']+'/rl-agent'
    try:
        with TestClient(create_app(core,{'a':'alice'})) as client:
            assert client.post(path,headers={'Authorization':'Bearer a'}).json()['error']=='AGENT_NOT_CONFIGURED'
        with TestClient(create_app(core,{'a':'alice','b':'bob'},rl_agent=agent)) as client:
            assert client.post(path).status_code==401
            assert client.post(path,headers={'Authorization':'Bearer b'}).status_code==404
            client.headers['Authorization']='Bearer a'
            assert client.post(path,json={}).json()['error']=='CLIENT_BODY_FORBIDDEN'
            assert client.post(path,headers={'Origin':'null'}).status_code==403
            result=client.post(path)
            assert result.status_code==200 and result.json()['policy_available']
            assert client.post(path).json()==result.json()
            status=client.get('/mcp/status').json()
            assert status['rl_mcp_configured'] and status['rl_entry_enabled'] and not status['real_rl_configured']
    finally:agent.close()


@pytest.mark.parametrize('fault',['wrong_case','origin','producer','version','hash'])
def test_invalid_parent_prediction_is_rejected_before_key(core,run,tmp_path,fault):
    core.execute_prediction('alice',run['id']);allow_data(core,run,'rl')
    with core.store.tx() as db:
        row=db.execute("SELECT * FROM artifacts WHERE kind='prediction'").fetchone()
        body=json.loads(row['body'])
        if fault=='hash': db.execute('UPDATE artifacts SET digest=? WHERE id=?',('a'*64,row['id']))
        else:
            body[{'wrong_case':'case_id','origin':'origin','producer':'producer','version':'version'}[fault]] = 'model' if fault=='origin' else 'a'*32
            db.execute('UPDATE artifacts SET body=?,digest=? WHERE id=?',(canonical(body),digest(body),row['id']))
    agent=RLAgent(core,tmp_path/'unread.key',tmp_path/'uncreated')
    try:
        with pytest.raises(GateError):agent.run('alice',run['id'])
        assert not agent.data_dir.exists() and counts(core)['jobs']==1
    finally:agent.close()


@pytest.mark.parametrize('fault',['cancel','withdraw','timeout'])
def test_actual_sdk_and_hanging_rl_worker_reaped(tmp_path,snapshot,fault):
    from test_workers import ProbeRunner,alive
    numerical=ProbeRunner('hang',tmp_path/'rl-worker.json')
    core=Core(Store(tmp_path/'hang.sqlite3'),enable_fixtures=True,executor_timeout=.5 if fault=='timeout' else 5)
    case=core.create_case('alice',snapshot);run=core.start_run('alice',case['case_id'],'eval','hang')
    core.execute_prediction('alice',run['id']);core._runner.close();core._runner=numerical
    calls=[]
    def provider(req):
        calls.append(req)
        content=({'type':'tool_use','id':'start-rl','name':TOOL_PREFIX+'request_rl','input':{}}
            if len(calls)==1 else {'type':'text','text':'stopped'})
        return httpx.Response(200,stream=SSE(message_events(content)))
    agent=setup(core,run,tmp_path,provider=provider)
    try:
        with ThreadPoolExecutor() as pool:
            future=pool.submit(agent.run,'alice',run['id'])
            deadline=time.monotonic()+10
            while not numerical.path.exists() and time.monotonic()<deadline:time.sleep(.01)
            assert numerical.path.exists()
            pid=json.loads(numerical.path.read_text())['pid']
            if fault=='cancel':core.cancel('alice',run['id'])
            elif fault=='withdraw':
                core.data_permissions.update('alice',case['case_id'],{'snapshot_id':case['snapshot_id'],'purpose':'rl','allowed':False,'policy_version':'engineering-data-policy-v1'})
            with pytest.raises(GateError):future.result(timeout=15)
        assert not alive(pid) and not numerical._active and not agent.runner._active
        assert counts(core)['artifacts']==1 and counts(core)['releases']==0
        with core.store.tx() as db:assert db.execute("SELECT count(*) FROM sdk_sessions WHERE state!='REMOVED'").fetchone()[0]==0
        (tmp_path/'rl-fault-proof.json').write_text(canonical({'actual_sdk':True,'numeric_probe':'hang','fault':fault,
            'pid':pid,'workers_reaped':True,'sdk_directory_removed':True,'policy_artifacts':0,'paid_calls':0}))
    finally:agent.close();core.close()


def test_actual_http_prediction_then_rl_requires_separate_authorization(tmp_path,snapshot):
    import socket
    from threading import Thread
    import uvicorn
    core=Core(Store(tmp_path/'http.sqlite3'),enable_fixtures=True)
    pr,pref,rr,rref=[],[],[],[]
    key=tmp_path/'offline.key';key.write_text('offline-http');key.chmod(0o600)
    from medical_harness.prediction_agent import PredictionAgent
    pred=PredictionAgent(core,key,tmp_path/'pred',transport=httpx.MockTransport(provider_for(PREDICTION_OPERATIONS,pr,pref,repeat=False)))
    rl=RLAgent(core,key,tmp_path/'rl',transport=httpx.MockTransport(provider_for(RL_OPERATIONS,rr,rref,repeat=False)))
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(create_app(core,{'offline-token':'alice'},prediction_agent=pred,rl_agent=rl),access_log=False,log_level='error',ws='none'))
    thread=Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    try:
        deadline=time.monotonic()+5
        while not server.started and time.monotonic()<deadline:time.sleep(.02)
        assert server.started
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',trust_env=False,timeout=20,headers={'Authorization':'Bearer offline-token'}) as client:
            case=client.post('/cases',json=snapshot).json()
            run=client.post('/runs',json={'case_id':case['case_id'],'expected_snapshot_id':case['snapshot_id'],'mode':'eval','idempotency_key':uuid4().hex}).json()
            path='/runs/'+run['id']
            permission={'snapshot_id':case['snapshot_id'],'allowed':True,'policy_version':'engineering-data-policy-v1'}
            assert client.post(path+'/rl-agent').json()['error']=='DATA_PERMISSION_REQUIRED' and not rr
            client.put('/cases/'+case['case_id']+'/data-permissions',json=dict(permission,purpose='prediction'))
            assert client.post(path+'/prediction-agent').json()['prediction_available']
            assert client.post(path+'/rl-agent').json()['error']=='DATA_PERMISSION_REQUIRED' and not rr
            client.put('/cases/'+case['case_id']+'/data-permissions',json=dict(permission,purpose='rl'))
            result=client.post(path+'/rl-agent')
            assert result.status_code==200 and result.json()['policy_available']
            assert client.post(path+'/rl-agent').json()==result.json() and len(rr)==3
            assert counts(core)=={'jobs':4,'artifacts':2,'drafts':0,'reviews':0,'releases':0}
            (tmp_path/'rl-http-proof.json').write_text(canonical({'actual_http':True,'actual_sdk_mcp':True,
                'separate_prediction_rl_permissions':True,'http_retry_reuses':True,'prediction_requests':len(pr),
                'rl_requests':len(rr),'counts':counts(core),'paid_calls':0}))
    finally:server.should_exit=True;thread.join(5);sock.close();pred.close();rl.close();core.close()
    assert not thread.is_alive()


def test_audit_failure_cannot_complete_rl(core,run,tmp_path,monkeypatch):
    core.execute_prediction('alice',run['id'])
    original=core._event
    def event(db,run_id,kind,**details):
        if kind=='rl_agent_completed':raise sqlite3.OperationalError('private-log-content')
        return original(db,run_id,kind,**details)
    monkeypatch.setattr(core,'_event',event)
    agent=setup(core,run,tmp_path,action=success)
    try:
        with pytest.raises(GateError,match='RL_AGENT_FAILED'):agent.run('alice',run['id'])
        assert not core.status('alice',run['id'])['currently_valid']
        assert 'private-log-content' not in canonical(agent.evidence)
        with pytest.raises(GateError):core.create_draft_job('alice',run['id'])
    finally:agent.close()
