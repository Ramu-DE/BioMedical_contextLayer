"""HTTP server for the context layer. Task 7.4.

Matches the module-5 MicroVM contract exactly:
  - agent on 8080:  POST /invoke {"question": "..."} -> GovernedResponse
                    GET  /health
  - hooks on 9000:  POST /aws/lambda-microvms/runtime/v1/ready

Credentials come from the execution role, so no keys are baked into the image.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from context_layer.agent.pipeline import ContextLayerAgent  # noqa: E402

APP_PORT = int(os.environ.get("APP_PORT", "8080"))
HOOKS_PORT = int(os.environ.get("HOOKS_PORT", "9000"))

_ready = threading.Event()
_agent: ContextLayerAgent | None = None
_lock = threading.Lock()


def agent() -> ContextLayerAgent:
    global _agent
    with _lock:
        if _agent is None:
            _agent = ContextLayerAgent(config_module.load())
        return _agent


class AgentHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[context-layer] {self.address_string()} {fmt % args}")

    def _json(self, status, body):
        payload = json.dumps(body, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            # This is the JSON API, not the dashboard. Say so rather than
            # returning a bare 404, which reads as "the app is broken".
            self._json(200, {
                "service": "context-layer API",
                "note": "This is the JSON API. The dashboard is a separate process.",
                "dashboard": "run ./run-ui.sh, then open the URL it prints",
                "endpoints": {
                    "GET /health": "readiness and config problems",
                    "POST /invoke": '{"question": "..."} -> GovernedResponse',
                },
            })
        elif self.path == "/health":
            problems = config_module.load().check()
            self._json(
                200 if not problems else 503,
                {"status": "ok" if not problems else "not_ready", "problems": problems},
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/invoke":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"invalid JSON: {e}"})
            return
        question = body.get("question") or body.get("message")
        if not question:
            self._json(400, {"error": "missing 'question' field"})
            return
        try:
            trace = agent().run(question)
            payload = trace.response.to_dict()
            payload["decision_id"] = trace.decision_id
            payload["timings_ms"] = {k: round(v) for k, v in trace.timings.items()}
            self._json(200, payload)
        except Exception as e:  # noqa: BLE001
            print(f"[context-layer] error: {e!r}")
            self._json(500, {"error": str(e)})


class HooksHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.path == "/aws/lambda-microvms/runtime/v1/ready":
            self.send_response(200 if _ready.is_set() else 503)
        else:
            self.send_response(200)
        self.end_headers()


def serve(port, handler):
    ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()


def main():
    threading.Thread(target=serve, args=(HOOKS_PORT, HooksHandler), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", APP_PORT), AgentHandler)
    cfg = config_module.load()
    print(f"context layer on :{APP_PORT} (model={cfg.bedrock_model_id}, "
          f"vector={cfg.vector_backend}, region={cfg.aws_region}), hooks on :{HOOKS_PORT}")
    _ready.set()
    server.serve_forever()


if __name__ == "__main__":
    main()
