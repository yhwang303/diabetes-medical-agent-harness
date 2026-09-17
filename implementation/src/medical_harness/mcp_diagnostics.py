"""Manual offline SDK/MCP probe, isolated from all user cases and real credentials."""

import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

import httpx

from .contracts import GateError, digest
from .core import Core
from .data_policy import POLICY_VERSION
from .paths import ROOT, confined
from .prediction_agent import PredictionAgent
from .sdk_worker import MODEL, SDK_VERSION, PREDICTION_OPERATIONS, RL_OPERATIONS, TOOL_PREFIX, prediction_configuration
from .store import Store


class OfflineReply(httpx.SyncByteStream):
    """Fixed local protocol replies exercise the SDK's real MCP tool transport."""
    def __init__(self, content):
        tool = content['type'] == 'tool_use'
        self.events = [
            {'type': 'message_start', 'message': {'id': uuid4().hex, 'type': 'message', 'role': 'assistant',
                'model': MODEL, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 10, 'output_tokens': 0}}},
            {'type': 'content_block_start', 'index': 0, 'content_block': dict(content, input={}) if tool else {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(content['input'])}
                if tool else {'type': 'text_delta', 'text': content['text']}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use' if tool else 'end_turn', 'stop_sequence': None},
                'usage': {'output_tokens': 1}},
            {'type': 'message_stop'}]

    def __iter__(self):
        for event in self.events:
            yield ('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n').encode()


def tool_result(body):
    for message in reversed(body['messages']):
        if not isinstance(message.get('content'), list): continue
        for item in reversed(message['content']):
            if item.get('type') != 'tool_result': continue
            content = item['content']
            return json.loads(content if isinstance(content, str) else '\n'.join(x.get('text', '') for x in content))
    raise GateError('MCP_CHECK_PROTOCOL')


def offline_prediction_check(directory):
    core = Core(Store(directory / 'check.sqlite3'), enable_fixtures=True)
    agent = None
    try:
        case = core.create_case('mcp-self-check', json.loads((ROOT / 'examples/synthetic-case.json').read_text()))
        core.data_permissions.update('mcp-self-check', case['case_id'], {
            'snapshot_id': case['snapshot_id'], 'purpose': 'prediction', 'allowed': True, 'policy_version': POLICY_VERSION})
        run = core.start_run('mcp-self-check', case['case_id'], 'eval', 'local-check')
        key = directory / 'offline.key'
        with open(key, 'x', opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
            stream.write('synthetic-offline-key-no-account')
        calls, references = [], []
        def provider(request):
            body = json.loads(request.content)
            calls.append(body)
            if {t['name'] for t in body['tools']} != {TOOL_PREFIX + name for name in PREDICTION_OPERATIONS}:
                raise GateError('MCP_CHECK_TOOLSET')
            phase = len(calls)
            if phase == 1:
                name, arguments = 'request_prediction', {}
            elif phase in (2, 3):
                previous = tool_result(body)
                if not previous.get('prediction_available') or previous.get('evidence', {}).get('origin') != 'fixture':
                    raise GateError('MCP_CHECK_EVIDENCE')
                references.append(previous)
                if phase == 3:
                    if references[0] != references[1]: raise GateError('MCP_CHECK_EVIDENCE')
                    return httpx.Response(200, stream=OfflineReply({'type': 'text', 'text': 'Local synthetic check complete.'}))
                name, arguments = 'inspect_prediction', {'job_id': previous['job_id']}
            else:
                raise GateError('MCP_CHECK_PROTOCOL')
            return httpx.Response(200, stream=OfflineReply({'type': 'tool_use', 'id': 'check-' + str(phase),
                'name': TOOL_PREFIX + name, 'input': arguments}))
        agent = PredictionAgent(core, key, directory / 'agent', transport=httpx.MockTransport(provider), timeout=20)
        receipt = agent.run('mcp-self-check', run['id'])
        if len(calls) != 3 or len(references) != 2 or receipt['evidence'] != references[0]['evidence']:
            raise GateError('MCP_CHECK_INCOMPLETE')
    finally:
        if agent: agent.close()
        core.close()


class McpDiagnostics:
    def __init__(self, core, *, prediction_enabled=False, rl_enabled=False):
        self.core, self.prediction_enabled = core, prediction_enabled
        self.rl_enabled = rl_enabled
        self._check_lock, self._state_lock = Lock(), Lock()
        self._results = {}

    def status(self, owner):
        try: installed = version('claude-agent-sdk')
        except PackageNotFoundError: installed = None
        with self._state_lock:
            result = self._results.get(owner, {'state': 'NOT_CHECKED', 'checked_at': None, 'error': None})
            return {'sdk_version': installed, 'expected_sdk_version': SDK_VERSION,
                    'tools': list(PREDICTION_OPERATIONS), 'configuration_hash': digest(prediction_configuration()),
                    'prediction_entry_enabled': self.prediction_enabled, 'real_prediction_configured': False,
                    'rl_mcp_configured': True, 'rl_tools': list(RL_OPERATIONS),
                    'rl_entry_enabled': self.rl_enabled, 'real_rl_configured': False, 'online_provider_checked': False,
                    'scope': 'offline_synthetic', 'last_check': dict(result)}

    def check(self, owner):
        if not self._check_lock.acquire(blocking=False): raise GateError('MCP_CHECK_BUSY', 409)
        started = time.monotonic()
        result = {'state': 'RUNNING', 'checked_at': None, 'error': None}
        try:
            with self._state_lock: self._results[owner] = dict(result)
            with self.core.store.tx() as db:
                self.core._event(db, None, 'mcp_check_started', owner=owner, scope='offline_synthetic')
            try:
                if self.status(owner)['sdk_version'] != SDK_VERSION: raise GateError('MCP_CHECK_SDK_VERSION')
                parent = confined(Path(self.core.store.path).parent / 'mcp-checks')
                parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with TemporaryDirectory(prefix='check-', dir=parent) as temporary:
                    offline_prediction_check(Path(temporary))
                result['state'] = 'PASSED'
            except Exception as exc:
                result.update(state='FAILED', error=exc.code if isinstance(exc, GateError) else 'MCP_CHECK_FAILED')
            result.update(checked_at=time.time(), duration_ms=int((time.monotonic() - started) * 1000))
            with self.core.store.tx() as db:
                self.core._event(db, None, 'mcp_check_finished', owner=owner, scope='offline_synthetic', **result)
            return dict(self.status(owner), last_check=dict(result))
        except Exception:
            result.update(state='FAILED', checked_at=time.time(), error='MCP_CHECK_AUDIT_FAILED')
            raise
        finally:
            with self._state_lock: self._results[owner] = dict(result)
            self._check_lock.release()
