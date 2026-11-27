"""Tiny job server that runs ON a rented GPU pod (started by experiments/runpod_driver.py).

It lets the driver start shell jobs in the repository checkout, read their logs and fetch
result files over the pod's HTTPS proxy.  Every request must carry the token passed to the
pod in the environment variable POD_TOKEN.  Standard library only.

    GET  /health
    POST /run     {"name": "...", "cmd": "..."}      start a job (bash -lc) in the background
    GET  /jobs                                       name, pid, exit code of every job
    GET  /log?name=...&tail=20000                    end of a job's log
    POST /kill    {"name": "..."}
    GET  /file?path=results/...                      a file below the checkout (or logs/)
    GET  /ls?path=results/...
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
TOKEN = os.environ.get("POD_TOKEN", "")
JOBS: dict = {}


def safe(path: str) -> pathlib.Path:
    p = (ROOT / path).resolve()
    if ROOT not in p.parents and p != ROOT:
        raise PermissionError(path)
    return p


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # quiet
        pass

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self) -> bool:
        if not TOKEN or self.headers.get("x-token") != TOKEN:
            self._send(403, {"error": "forbidden"})
            return False
        return True

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/health":
            return self._send(200, {"ok": True})
        if not self._auth():
            return
        try:
            if u.path == "/jobs":
                return self._send(200, {n: dict(pid=p.pid, exit=p.poll(), cmd=c) for n, (p, c) in JOBS.items()})
            if u.path == "/log":
                f = LOGS / f"{q['name']}.log"
                data = f.read_bytes() if f.exists() else b""
                return self._send(200, data[-int(q.get("tail", 20000)):], "text/plain")
            if u.path == "/file":
                return self._send(200, safe(q["path"]).read_bytes(), "application/octet-stream")
            if u.path == "/ls":
                p = safe(q.get("path", "."))
                return self._send(200, sorted((str(x.relative_to(ROOT)), x.stat().st_size) for x in p.rglob("*") if x.is_file()))
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": repr(e)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth():
            return
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        u = urlparse(self.path)
        try:
            if u.path == "/run":
                name = body["name"]
                if name in JOBS and JOBS[name][0].poll() is None:
                    return self._send(409, {"error": f"job {name} is still running"})
                LOGS.mkdir(exist_ok=True)
                log = open(LOGS / f"{name}.log", "ab")
                p = subprocess.Popen(["bash", "-lc", body["cmd"]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                JOBS[name] = (p, body["cmd"])
                return self._send(200, {"pid": p.pid})
            if u.path == "/kill":
                p = JOBS[body["name"]][0]
                p.kill()
                return self._send(200, {"killed": p.pid})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": repr(e)})
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"pod server on :{port}, root {ROOT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
