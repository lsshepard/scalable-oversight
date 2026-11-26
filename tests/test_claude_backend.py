"""The Claude backend against a local stand-in for the Messages API (no network, no key).

Checks the request the official SDK actually sends, that ambient ANTHROPIC_* settings of
a surrounding session are never used, and how replies, refusals and a rejected key are
mapped."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("anthropic")

from hpo.claude_backend import API_URL, KEY_ENV, PRICES, claude_backend  # noqa: E402
from hpo.rounds import FatalBackendError, GameConfig, play_trajectory  # noqa: E402

OK = dict(id="msg_1", type="message", role="assistant", model="claude-sonnet-5-5",
          content=[dict(type="thinking", thinking="one line is wrong", signature="s"),
                   dict(type="text", text="Line 3 is wrong.\nREPORT: 1\nNOTE: stay accurate")],
          stop_reason="end_turn", stop_sequence=None,
          usage=dict(input_tokens=900, output_tokens=350))
REFUSAL = dict(OK, content=[], stop_reason="refusal",
               stop_details=dict(type="refusal", category="general_harms", explanation="x"))


@pytest.fixture
def api(monkeypatch):
    """Fake Messages endpoint; api.reply is (status, body), api.seen collects requests."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            server.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            status, payload = server.reply
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *a):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.seen, server.reply = [], (200, OK)
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    yield server
    server.shutdown()


def test_request_shape_reply_mapping_and_no_ambient_credentials(api, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "experiment-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")       # nothing listens here
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Ambient: ambient-header")
    complete = claude_backend("claude-sonnet-5-5", effort="medium", base_url=api.url, max_retries=0)
    rep = complete("SYSTEM", "USER")

    assert rep.text.endswith("NOTE: stay accurate") and rep.thinking == "one line is wrong"
    assert (rep.stop_reason, rep.in_tokens, rep.out_tokens) == ("end_turn", 900, 350)

    (path, headers, body), = api.seen
    assert path == "/v1/messages"
    assert body == dict(model="claude-sonnet-5-5", max_tokens=16000, system="SYSTEM",
                        messages=[dict(role="user", content="USER")],
                        thinking=dict(type="adaptive", display="summarized"),
                        output_config=dict(effort="medium"))             # no sampling parameters
    sent = json.dumps(headers)
    assert "experiment-key" in sent and "ambient" not in sent


def test_effort_is_omitted_unless_set(api, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "k")
    claude_backend("claude-haiku-5-5", base_url=api.url, max_retries=0)("S", "U")
    assert "output_config" not in api.seen[0][2]


def test_key_comes_from_the_experiment_variable_only(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    with pytest.raises(FatalBackendError, match=KEY_ENV):
        claude_backend("claude-sonnet-5-5")
    assert API_URL == "https://api.anthropic.com"
    assert claude_backend.__defaults__[3] == API_URL                      # default base_url
    assert set(PRICES) >= {"claude-sonnet-5-5", "claude-haiku-5-5"}


def test_refusal_is_recorded_and_voids_the_round(api, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "k")
    api.reply = (200, REFUSAL)
    complete = claude_backend("claude-sonnet-5-5", base_url=api.url, max_retries=0)
    rep = complete("S", "U")
    assert rep.text == "" and rep.stop_reason == "refusal"
    recs = play_trajectory(GameConfig(rounds=1, parse_retries=0), complete, seed=0)
    assert all(r["void"] and not r["parsed"] and r["stop_reason"] == "refusal" for r in recs)


def test_rejected_key_is_fatal(api, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "bad")
    api.reply = (401, dict(type="error", error=dict(type="authentication_error", message="invalid x-api-key")))
    with pytest.raises(FatalBackendError, match="rejected the key"):
        claude_backend("claude-sonnet-5-5", base_url=api.url, max_retries=0)("S", "U")
