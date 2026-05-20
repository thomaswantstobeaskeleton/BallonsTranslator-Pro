from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Any
from urllib.parse import parse_qs, urlparse

from utils.automation_api_contract import build_route_discovery


class LocalAutomationApiServer:
    def __init__(self, host: str, port: int, handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]], api_key: str = ""):
        self.host = host
        self.port = int(port)
        self.handlers = handlers
        self._thread = None
        self._server = None
        self.api_key = (api_key or "").strip()

    def start(self):
        if self._thread is not None:
            return
        handlers = self.handlers

        def _routes_payload():
            return build_route_discovery(handlers)

        class H(BaseHTTPRequestHandler):
            def _auth_ok(self):
                if self.server_ref.api_key:
                    got = (self.headers.get('X-API-Key', '') or '').strip()
                    if got != self.server_ref.api_key:
                        self._send(401, {'ok': False, 'error': 'unauthorized'})
                        return False
                return True

            def do_GET(self):
                parsed = urlparse(self.path)
                route = parsed.path.strip('/')
                if route in {'', 'health'}:
                    if not self._auth_ok():
                        return
                    self._send(200, {**_routes_payload(), 'status': 'ok'})
                    return
                if route == 'routes':
                    if not self._auth_ok():
                        return
                    self._send(200, _routes_payload())
                    return
                if route == 'mcp/commands':
                    if not self._auth_ok():
                        return
                    payload = _routes_payload()
                    self._send(200, {
                        'ok': True,
                        'commands': payload.get('mcp_routes', []),
                        'count': len(payload.get('mcp_routes', [])),
                    })
                    return
                if route == 'events':
                    if not self._auth_ok():
                        return
                    self._send_events(parsed.query)
                    return
                self._send(404, {'ok': False, 'error': f'unknown route: {route}'})

            def do_POST(self):
                ln = int(self.headers.get('Content-Length', '0') or 0)
                raw = self.rfile.read(ln) if ln > 0 else b'{}'
                try:
                    body = json.loads(raw.decode('utf-8')) if raw else {}
                except Exception:
                    body = {}
                route = self.path.strip('/')
                if not self._auth_ok():
                    return
                fn = handlers.get(route)
                if fn is None:
                    self._send(404, {'ok': False, 'error': f'unknown route: {route}'})
                    return
                try:
                    out = fn(body if isinstance(body, dict) else {}) or {}
                    self._send(200, {'ok': True, 'result': out})
                except Exception as e:
                    self._send(500, {'ok': False, 'error': str(e)})

            def log_message(self, *args, **kwargs):
                return

            def _send(self, code: int, obj: Dict[str, Any]):
                data = json.dumps(obj).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_events(self, query: str):
                qs = parse_qs(query or '')
                job_id = (qs.get('job_id') or [''])[0]
                payload: Dict[str, Any] = {'ok': True, 'job_id': job_id, 'event': 'job_snapshot'}
                status_fn = handlers.get('job_status')
                logs_fn = handlers.get('job_logs')
                if job_id and status_fn is not None:
                    try:
                        payload['status'] = status_fn({'job_id': job_id}) or {}
                    except Exception as e:
                        self._send(404, {'ok': False, 'error': str(e)})
                        return
                if job_id and logs_fn is not None:
                    try:
                        payload['logs'] = logs_fn({'job_id': job_id}) or {}
                    except Exception as e:
                        payload['warnings'] = [str(e)]
                status_name = str(((payload.get('status') or {}).get('status', 'snapshot')) if isinstance(payload.get('status'), dict) else 'snapshot')
                event_id = str(((payload.get('status') or {}).get('updated_at', '0')) if isinstance(payload.get('status'), dict) else '0')
                lines = [
                    f"id: {event_id}",
                    f"event: {status_name}",
                    'data: ' + json.dumps(payload),
                    '',
                ]
                data = ('\n'.join(lines) + '\n').encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        H.server_ref = self

        self._server = ThreadingHTTPServer((self.host, self.port), H)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None
