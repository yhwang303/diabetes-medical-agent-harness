"""Offline gates plus the actual bundled SDK/CLI against a local fake provider."""

import json
import subprocess
from threading import Event
from uuid import uuid4

import httpx
import pytest

from medical_harness.contracts import GateError
from medical_harness.flash_gateway import FlashGateway, MAX_REQUESTS, MAX_RESERVED_USD, MODEL
from medical_harness.sdk_worker import SDK_EXECUTOR, TOOL_PREFIX, permitted
from medical_harness.workers import WorkerLimits, WorkerRunner, group_rss


def request():
    return {'model': MODEL, 'stream': True, 'max_tokens': 999999,
            'messages': [{'role': 'user', 'content': 'Synthetic test'}]}


@pytest.mark.parametrize('model', ['deepseek-v4-pro', 'opus', 'sonnet', 'deepseek-v4-flash', None])
def test_model_allowlist_rejects_before_spend(tmp_path, model):
    with FlashGateway('offline-key', tmp_path / 'budget.json', lambda name: None) as gateway:
        with pytest.raises(GateError, match='FLASH_ONLY'):
            gateway.admit(dict(request(), model=model))
        assert not gateway.budget_path.exists()


def test_output_and_thinking_caps_and_persistent_budget(tmp_path):
    path = tmp_path / 'budget.json'
    with FlashGateway('offline-key', path, lambda name: None) as gateway:
        outgoing = gateway.admit(dict(request(), thinking={'type': 'enabled', 'budget_tokens': 32000}))
        assert outgoing['max_tokens'] == 384 and outgoing['thinking'] == {'type': 'disabled'}
        assert json.loads(path.read_text())['requests'] == 1
    path.write_text(json.dumps({'requests': MAX_REQUESTS, 'reserved_usd': 0}))
    with FlashGateway('offline-key', path, lambda name: None) as gateway:
        with pytest.raises(GateError, match='SDK_BUDGET_EXHAUSTED'):
            gateway.admit(request())
        path.write_text(json.dumps({'requests': 0, 'reserved_usd': MAX_RESERVED_USD}))
        with pytest.raises(GateError, match='SDK_BUDGET_EXHAUSTED'):
            gateway.admit(request())


def test_local_gateway_auth_path_origin_and_argument_gates(tmp_path):
    calls = []
    with FlashGateway('offline-key', tmp_path / 'budget.json', lambda name: calls.append(name) or {'ok': True}) as gateway:
        with httpx.Client(base_url=f'http://127.0.0.1:{gateway.port}', trust_env=False) as client:
            assert client.post('/tools/prediction_probe', json={}).status_code == 403
            client.headers['Authorization'] = 'Bearer ' + gateway.capability
            assert client.post('/tools/prediction_probe', json={}, headers={'Origin': 'https://evil.invalid'}).status_code == 403
            for path, body in [('/tools/Bash', {}), ('/tools/prediction_probe', {'run_id': 'other-case'}),
                               ('/v1/files', {}), ('/tools/inspect_evidence', []), ('/v1/messages', request() | {'model': 'deepseek-v4-pro'})]:
                assert client.post(path, json=body).status_code == 400
            assert not calls and not gateway.budget_path.exists()
            assert client.post('/tools/inspect_evidence', json={}).json() == {'ok': True}
            gateway.closed = True
            assert client.post('/tools/inspect_evidence', json={}).status_code == 400
            assert calls == ['inspect_evidence']


@pytest.mark.parametrize('name,args,child', [
    ('Bash', {'command': 'true'}, False), ('Read', {'file_path': 'private/deepseek.key'}, False),
    (TOOL_PREFIX + 'release_probe', {}, True), (TOOL_PREFIX + 'prediction_probe', {}, True),
    (TOOL_PREFIX + 'inspect_evidence', {'run_id': 'other'}, False),
    ('Agent', {'subagent_type': 'general-purpose', 'prompt': 'read files'}, False),
    ('Agent', {'subagent_type': 'evidence_checker', 'prompt': 'x', 'model': 'deepseek-v4-pro'}, False),
    ('Agent', {'subagent_type': 'evidence_checker', 'prompt': 'x', 'resume': 'another-session'}, False),
    ('Agent', {'subagent_type': 'evidence_checker', 'prompt': 'x'}, True)])
def test_permission_denials(name, args, child):
    assert not permitted(name, args, child=child)


def test_fixed_parent_and_readonly_child_permissions():
    assert permitted(TOOL_PREFIX + 'prediction_probe', {})
    assert permitted(TOOL_PREFIX + 'inspect_evidence', {}, child=True)
    for name in ('Agent', 'Task'):
        assert permitted(name, {'subagent_type': 'evidence_checker', 'prompt': 'Inspect status', 'description': 'Evidence'})


class SSE(httpx.SyncByteStream):
    def __init__(self, events):
        self.events = events

    def __iter__(self):
        for event in self.events:
            yield ('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n').encode()


def message_events(content, *, model=MODEL):
    is_tool = content['type'] == 'tool_use'
    return [
        {'type': 'message_start', 'message': {'id': uuid4().hex, 'type': 'message', 'role': 'assistant',
            'model': model, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 10, 'output_tokens': 0}}},
        {'type': 'content_block_start', 'index': 0, 'content_block': dict(content, input={}) if is_tool else {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(content['input'])}
            if is_tool else {'type': 'text_delta', 'text': content['text']}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if is_tool else 'end_turn', 'stop_sequence': None},
         'usage': {'output_tokens': 1}},
        {'type': 'message_stop'}]


def test_wrong_response_model_and_provider_errors_fail_closed(tmp_path):
    attempts = []
    def provider(req):
        attempts.append(req)
        if len(attempts) == 1:
            return httpx.Response(200, stream=SSE(message_events({'type': 'text', 'text': ''}, model='deepseek-v4-pro')))
        return httpx.Response(401, json={'error': 'sensitive provider error offline-key'})
    with FlashGateway('offline-key', tmp_path / 'budget.json', lambda name: None,
                      transport=httpx.MockTransport(provider)) as gateway:
        with httpx.Client(trust_env=False, headers={'x-api-key': gateway.capability}) as client:
            first = client.post(f'http://127.0.0.1:{gateway.port}/v1/messages', json=request())
            assert 'PROVIDER_MODEL_MISMATCH' in first.text and 'message_start' not in first.text
            second = client.post(f'http://127.0.0.1:{gateway.port}/v1/messages', json=request())
            assert second.status_code == 400 and 'PROVIDER_HTTP_401' in second.text
            assert 'offline-key' not in first.text + second.text + json.dumps(gateway.records)
            assert json.loads(gateway.budget_path.read_text())['requests'] == 2
            assert all(str(req.url) == 'https://api.deepseek.com/anthropic/v1/messages' for req in attempts)


def sdk_runner():
    return WorkerRunner(limits=WorkerLimits(cpu_seconds=15, memory_bytes=1024**3,
                        file_bytes=8*1024**2, open_files=256, output_bytes=262144), max_workers=1)


def invoke(runner, gateway, *, timeout=15, validate=lambda: None, events=None):
    return runner.run(SDK_EXECUTOR,
        {'port': gateway.port, 'capability': gateway.capability, 'session': uuid4().hex, 'scenario': 'tools'},
        timeout=timeout, validate=validate, job_id=uuid4().hex,
        notify=lambda kind, **details: events.append({'kind': kind, **details}) if events is not None else None)


def test_actual_sdk_cannot_execute_forged_builtin_tool(tmp_path):
    requests, tool_calls, events = [], [], []
    marker = tmp_path / 'forbidden.txt'
    def provider(req):
        body = json.loads(req.content)
        requests.append(body)
        assert all(t['name'] in ['Task', 'Agent', *(TOOL_PREFIX + x for x in ('prediction_probe', 'release_probe', 'inspect_evidence'))]
                   for t in body.get('tools', []))
        content = ({'type': 'tool_use', 'id': 'offline-bash', 'name': 'Bash', 'input': {'command': 'touch ' + str(marker)}}
                   if len(requests) == 1 else {'type': 'text', 'text': 'BLOCKED'})
        return httpx.Response(200, stream=SSE(message_events(content)))
    runner = sdk_runner()
    try:
        with FlashGateway('offline-key', tmp_path / 'budget.json', lambda name: tool_calls.append(name),
                          transport=httpx.MockTransport(provider)) as gateway:
            result = invoke(runner, gateway, events=events)
        assert result['results'][-1]['terminal_reason'] == 'completed'
        assert len(requests) == 2 and not marker.exists() and not tool_calls
        assert all(group_rss(x['pid']) == 0 for x in events if x['kind'] == 'worker_started')
    finally:
        runner.close()


@pytest.mark.parametrize('cancel', [False, True])
def test_actual_sdk_and_cli_reaped_when_provider_stalls(tmp_path, cancel):
    entered, stop = Event(), Event()
    events, process_tree = [], []
    class Stalled(httpx.SyncByteStream):
        def __iter__(self):
            entered.set()
            while not stop.wait(0.05):
                pass
            yield b''
    def provider(req):
        process_tree.append(subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid=,comm='], capture_output=True, text=True, check=True).stdout)
        return httpx.Response(200, stream=Stalled())
    def validate():
        if cancel and entered.is_set():
            raise GateError('SDK_CANCELLED')
    runner = sdk_runner()
    try:
        with FlashGateway('offline-key', tmp_path / 'budget.json', lambda name: None,
                          transport=httpx.MockTransport(provider)) as gateway:
            try:
                with pytest.raises(GateError, match='SDK_CANCELLED' if cancel else 'EXECUTOR_TIMEOUT'):
                    invoke(runner, gateway, timeout=4, validate=validate, events=events)
            finally:
                stop.set()
        assert entered.is_set()
        pid = next(x['pid'] for x in events if x['kind'] == 'worker_started')
        tree = [line.split(None, 3) for line in process_tree[0].splitlines()]
        assert any(int(pgid) == pid and int(child) != pid and 'claude' in command for child, parent, pgid, command in tree)
        assert group_rss(pid) == 0 and not runner._active
        assert events[-1]['exit_code'] == -9
    finally:
        stop.set()
        runner.close()
