"""Web console (ADR-0012): API contract, chat proxy, static page."""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from astra.config import AstraConfig
from astra.ui.server import UiState, make_handler


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class FakeEngine(BaseHTTPRequestHandler):
    """Minimal llama-server: /v1/models and a streamed chat completion."""

    def do_GET(self) -> None:
        body = json.dumps({"data": [{"id": "qwen2.5-14b-q4"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert req["stream"] is True and req["messages"][-1]["content"] == "hi"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ("Hello", " from", " the pool"):
            chunk = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def engine() -> Iterator[str]:
    server, url = _serve(FakeEngine)
    yield url
    server.shutdown()
    server.server_close()


@pytest.fixture
def console(engine: str) -> Iterator[tuple[str, UiState]]:
    state = UiState(AstraConfig(), simulate="fabric-demo", engine_url=engine)
    server, url = _serve(make_handler(state))
    yield url, state
    server.shutdown()
    server.server_close()


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def _post(url: str, body: dict[str, Any]) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_index_is_self_contained(console: tuple[str, UiState]) -> None:
    url, _ = console
    with urllib.request.urlopen(url + "/", timeout=5) as r:
        html = r.read().decode()
    assert r.headers["Content-Type"].startswith("text/html")
    assert "<title>ASTRA Console</title>" in html
    assert "http://" not in html.replace("http://127.0.0.1", "")  # no external assets


def test_state_topology_and_engine(console: tuple[str, UiState]) -> None:
    url, _ = console
    s = _get(url + "/api/state")
    nodes = {n["name"]: n for n in s["pool"]["nodes"]}
    assert s["pool"]["simulated"] is True
    assert nodes["astra-01"]["role"] == "head" and nodes["desktop"]["role"] == "worker"
    assert nodes["desktop"]["gpus"][0]["rpc_endpoint"] == "desktop:50052"
    assert s["engine"] == {"url": s["engine"]["url"], "online": True, "model": "qwen2.5-14b-q4"}
    assert s["health"]["driver_window"] == [455, 580]  # GT 1030 on the head pins R580
    assert s["plan"] is None


def test_plan_becomes_active(console: tuple[str, UiState]) -> None:
    url, state = console
    status, body = _post(url + "/api/plan", {"model": "qwen2.5-14b", "quant": "q4_k_m"})
    plan = json.loads(body)
    assert status == 200 and plan["fits"]
    assert [d["node"] for d in plan["devices"]] == ["local", "desktop"]
    assert "--rpc desktop:50052" in plan["launch"]["command"]
    assert _get(url + "/api/state")["plan"]["layer_counts"] == plan["layer_counts"]
    assert state.active_plan is not None


@pytest.mark.parametrize(
    "body",
    [
        {"model": "nope"},
        {"model": "qwen2.5-14b", "quant": "awq-int4"},
        {"model": "qwen2.5-14b", "context": 10},
    ],
)
def test_plan_rejects_bad_input(console: tuple[str, UiState], body: dict[str, Any]) -> None:
    status, raw = _post(console[0] + "/api/plan", body)
    assert status == 400 and "error" in json.loads(raw)


def test_models_endpoint(console: tuple[str, UiState]) -> None:
    data = _get(console[0] + "/api/models")
    assert any(m["key"] == "qwen2.5-14b" for m in data["models"])
    assert {"key": "awq-int4", "engines": ["vllm"]} in data["quants"]


def test_chat_streams_through_proxy(console: tuple[str, UiState]) -> None:
    status, raw = _post(console[0] + "/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
    text = "".join(
        json.loads(line[5:])["choices"][0]["delta"]["content"]
        for line in raw.decode().splitlines()
        if line.startswith("data:") and "[DONE]" not in line
    )
    assert status == 200 and text == "Hello from the pool"


def test_chat_engine_offline_and_validation() -> None:
    state = UiState(AstraConfig(), simulate="reference", engine_url="http://127.0.0.1:1")
    server, url = _serve(make_handler(state))
    try:
        status, raw = _post(url + "/api/chat", {"messages": [{"role": "user", "content": "x"}]})
        assert status == 503 and "not reachable" in json.loads(raw)["error"]
        assert _post(url + "/api/chat", {"messages": []})[0] == 400
        assert _post(url + "/api/nope", {})[0] == 404
        assert _get(url + "/api/state")["engine"]["online"] is False
    finally:
        server.shutdown()
        server.server_close()
