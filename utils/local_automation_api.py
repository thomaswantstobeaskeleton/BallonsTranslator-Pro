from __future__ import annotations
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Any


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

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                ln = int(self.headers.get('Content-Length', '0') or 0)
                raw = self.rfile.read(ln) if ln > 0 else b'{}'
                try:
                    body = json.loads(raw.decode('utf-8')) if raw else {}
                except Exception:
                    body = {}
                route = self.path.strip('/')
                if self.server_ref.api_key:
                    got = (self.headers.get('X-API-Key', '') or '').strip()
                    if got != self.server_ref.api_key:
                        self._send(401, {'ok': False, 'error': 'unauthorized'})
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
