"""Local, run-scoped compatibility gateway. The real provider key stays in the host."""

import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import httpx

from .contracts import GateError, canonical, strict_json
from .sdk_worker import MODEL, OPERATIONS
from .paths import confined

MAX_REQUESTS = 12
MAX_RESERVED_USD = 0.05
MAX_OUTPUT_TOKENS = 384
UPSTREAM = 'https://api.deepseek.com/anthropic/v1/messages'


class FlashGateway:
    def __init__(self, key, budget_path, invoke_tool, *, transport=None, tool_handlers=None, authorize=None):
        self._key, self._tool = key, invoke_tool
        self.tool_handlers = tool_handlers
        self.authorize = authorize
        self.budget_path = confined(budget_path)
        self._lock, self._serial = Lock(), Lock()
        self.capability = secrets.token_urlsafe(32)
        self.records = []
        self.closed = False
        self.http = httpx.Client(trust_env=False, follow_redirects=False, timeout=25, transport=transport)
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                raw = canonical(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                self.connection.settimeout(30)
                self.streaming = False
                auth = self.headers.get('x-api-key') or self.headers.get('Authorization', '').removeprefix('Bearer ')
                if not hmac.compare_digest(auth.encode(), gateway.capability.encode()) or self.headers.get('Origin'):
                    self.reply(403, {'error': {'type': 'permission_error', 'message': 'GATEWAY_AUTH_REQUIRED'}})
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 262144 or self.headers.get('Transfer-Encoding'):
                        raise GateError('GATEWAY_INPUT_LIMIT')
                    body = strict_json(self.rfile.read(length))
                    with gateway._serial:
                        if gateway.closed:
                            raise GateError('SDK_SESSION_CLOSED')
                        if self.path.startswith('/tools/'):
                            name = self.path.removeprefix('/tools/')
                            if gateway.tool_handlers is not None:
                                if name not in gateway.tool_handlers or not isinstance(body, dict):
                                    raise GateError('SDK_TOOL_DENIED')
                                self.reply(200, gateway.tool_handlers[name](body))
                            else:
                                if name not in OPERATIONS or body != {}:
                                    raise GateError('SDK_TOOL_DENIED')
                                self.reply(200, gateway._tool(name))
                        elif self.path.split('?')[0] == '/v1/messages':
                            gateway.forward(self, body)
                        else:
                            raise GateError('GATEWAY_PATH_DENIED')
                except (GateError, ValueError, httpx.HTTPError) as exc:
                    code = exc.code if isinstance(exc, GateError) else 'GATEWAY_REQUEST_FAILED'
                    gateway.records.append({'error': code})
                    try:
                        error = {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': code}}
                        if self.streaming:
                            self.wfile.write(('event: error\ndata: ' + canonical(error) + '\n\n').encode())
                            self.wfile.flush()
                        else:
                            self.reply(400, error)
                    except OSError:
                        pass
                except OSError:
                    pass  # A cancelled SDK closes its stream. Never echo request bodies or credentials.
                except Exception:
                    gateway.records.append({'error': 'GATEWAY_INTERNAL_ERROR'})
                    try:
                        self.reply(500, {'error': {'type': 'api_error', 'message': 'GATEWAY_INTERNAL_ERROR'}})
                    except OSError:
                        pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self):
        return self.server.server_port

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.closed = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.http.close()

    def admit(self, body):
        if not isinstance(body, dict) or body.get('model') != MODEL:
            raise GateError('FLASH_ONLY')
        if body.get('stream') is not True:
            raise GateError('STREAM_REQUIRED')
        body = dict(body, max_tokens=MAX_OUTPUT_TOKENS, thinking={'type': 'disabled'})
        # Conservative reservation, not an invoice: doubled UTF-8 bytes plus protocol allowance,
        # full output cap, and peak cache-miss prices. No refund on errors or interruption.
        raw = canonical(body).encode()
        reserved = (2 * len(raw) + 4096) * 0.3 / 1_000_000 + MAX_OUTPUT_TOKENS * 1.2 / 1_000_000
        with self._lock:
            confined(self.budget_path)
            budget = json.loads(self.budget_path.read_text()) if self.budget_path.exists() else {'requests': 0, 'reserved_usd': 0.0}
            if budget['requests'] >= MAX_REQUESTS or budget['reserved_usd'] + reserved > MAX_RESERVED_USD:
                raise GateError('SDK_BUDGET_EXHAUSTED')
            budget['requests'] += 1
            budget['reserved_usd'] += reserved
            self.budget_path.write_text(canonical(budget))
            self.budget_path.chmod(0o600)
        return body

    def forward(self, handler, body):
        if self.authorize is not None:
            self.authorize()
        body = self.admit(body)
        record = {'model_requested': MODEL, 'model_responses': [], 'usage': [], 'status': None}
        self.records.append(record)
        with self.http.stream('POST', UPSTREAM, json=body,
                             headers={'x-api-key': self._key, 'anthropic-version': '2023-06-01'}) as upstream:
            record['status'] = upstream.status_code
            if upstream.status_code != 200:
                raise GateError('PROVIDER_HTTP_' + str(upstream.status_code))
            handler.send_response(200)
            handler.send_header('Content-Type', 'text/event-stream')
            handler.end_headers()
            handler.streaming = True
            size = 0
            frame = []
            seen_start = False
            for line in upstream.iter_lines():
                if self.authorize is not None:
                    self.authorize()
                if self.closed:
                    break
                size += len(line.encode()) + 1
                if size > 262144:
                    raise GateError('PROVIDER_OUTPUT_LIMIT')
                if line:
                    frame.append(line)
                    continue
                data = '\n'.join(x[5:].lstrip() for x in frame if x.startswith('data:'))
                if data:
                    event = strict_json(data)
                    message = event.get('message', {})
                    model = message.get('model')
                    if event.get('type') == 'error':
                        raise GateError('PROVIDER_STREAM_ERROR')
                    if event.get('type') == 'message_start':
                        if model != MODEL or seen_start:
                            raise GateError('PROVIDER_MODEL_MISMATCH')
                        seen_start = True
                        record['model_responses'].append(model)
                    elif not seen_start and event.get('type') != 'ping':
                        raise GateError('PROVIDER_PROTOCOL_ERROR')
                    for usage in (message.get('usage'), event.get('usage')):
                        if usage:
                            record['usage'].append({key: value for key, value in usage.items()
                                if key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens')
                                and type(value) is int and value >= 0})
                handler.wfile.write(('\n'.join(frame) + '\n\n').encode())
                handler.wfile.flush()
                frame = []
