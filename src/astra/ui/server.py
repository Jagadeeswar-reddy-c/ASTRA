"""``astra ui``: exo-style web console for the GPU pool (ADR-0012).

Serves one static page plus a small JSON API:

    GET  /                 the console (static HTML/CSS/JS, no external assets)
    GET  /api/state        pool topology + live GPU stats + active plan + health + engine status
    GET  /api/models       catalog models and quantizations
    POST /api/plan         plan a model on the pool; becomes the active plan shown on the map
    POST /api/chat         streamed chat completion, proxied to the configured engine only

Binds to 127.0.0.1 by default. The chat proxy only ever talks to the one engine
URL given at start-up, so the console cannot be used as an open proxy.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from astra import __version__, pool
from astra.config import AstraConfig
from astra.errors import AstraError
from astra.hardware.compat import analyse
from astra.planner.models import CATALOG, QUANTS, from_catalog
from astra.planner.split import plan as plan_on
from astra.planner.split import select_gpus

log = logging.getLogger("astra.ui")

MAX_BODY = 1 << 20
ENGINE_CHECK_S = 5.0


@dataclass
class UiState:
    config: AstraConfig
    simulate: str | None = None
    fabric: bool = False
    peers: str | None = None
    engine_url: str = "http://127.0.0.1:8080"
    active_plan: dict[str, Any] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _engine: tuple[float, bool, str | None] = (float("-inf"), False, None)

    # -- pool --------------------------------------------------------------------
    def _collect(self) -> pool.Pool:
        cfg = self.config
        return pool.collect(
            cfg,
            cfg.planner.gpu_memory_utilization,
            cfg.planner.reserve_mib,
            simulate=self.simulate,
            fabric=self.fabric or bool(self.peers),
            peers=self.peers,
            timeout_s=0.8,
        )

    def engine_status(self) -> tuple[bool, str | None]:
        checked, online, model = self._engine
        if time.monotonic() - checked < ENGINE_CHECK_S:
            return online, model
        online, model = False, None
        try:
            with urllib.request.urlopen(f"{self.engine_url}/v1/models", timeout=0.8) as resp:
                data = json.loads(resp.read())
                online = True
                models = data.get("data") or []
                model = models[0].get("id") if models else None
        except (urllib.error.URLError, OSError, ValueError):
            pass
        self._engine = (time.monotonic(), online, model)
        return online, model

    def state(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": __version__,
            "node": self.config.fabric.node_name or self.config.node.name,
            "time": time.time(),
        }
        try:
            p = self._collect()
            result["pool"] = p.to_dict()
            head = p.nodes[0] if p.nodes else None
            gpus = [g for g, _ in head.gpus] if head else []
            report = analyse(
                gpus,
                head.driver_version if head else None,
                self.config.chassis,
                None,
                self.config.interconnect.switch_vendor_ids,
                self.config.modules,
            )
            result["health"] = {
                "status": report.worst,
                "driver_window": list(report.required_driver_window),
                "cuda_architectures": report.cuda_architectures,
                "power": {
                    "psu_watts": report.power.psu_watts,
                    "sustained_watts": report.power.sustained_watts,
                    "peak_watts": report.power.peak_watts,
                    "limit_watts": report.power.sustained_limit_watts,
                    "recommended_psu_watts": report.power.recommended_psu_watts,
                    "ok": report.power.ok,
                },
                "findings": [f.__dict__ for f in report.findings],
            }
        except AstraError as exc:
            result["pool"] = {"simulated": bool(self.simulate), "nodes": [], "notes": [str(exc)]}
            result["health"] = None
        online, model = self.engine_status()
        result["engine"] = {"url": self.engine_url, "online": online, "model": model}
        with self._lock:
            result["plan"] = self.active_plan
        return result

    # -- planning ----------------------------------------------------------------
    def make_plan(self, req: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        model_key = str(req.get("model", ""))
        engine = str(req.get("engine", "llamacpp"))
        quant = str(req.get("quant") or ("awq-int4" if engine == "vllm" else "q4_k_m"))
        if model_key not in CATALOG:
            raise AstraError(f"unknown model '{model_key}'")
        if quant not in QUANTS or engine not in QUANTS[quant].engines:
            raise AstraError(f"quantization '{quant}' is not available for {engine}")
        context = int(req.get("context") or cfg.planner.default_context)
        if not 256 <= context <= 262144:
            raise AstraError("context must be between 256 and 262144 tokens")
        objective = str(req.get("objective") or cfg.planner.objective)
        profile = from_catalog(model_key, quant, str(req.get("kv_type", "f16")))
        p = self._collect()
        if not p.budgets:
            raise AstraError("no GPUs in the pool")
        util, hop = cfg.planner.gpu_memory_utilization, cfg.fabric.network_hop_ms / 1000
        selected = req.get("gpus")
        if isinstance(selected, list) and selected:
            chosen = [b for i in selected for b in p.budgets if b.index == int(i)]
            result = plan_on(
                profile, chosen, engine, context, util, p.notes, objective, network_hop_s=hop
            )
        else:
            result = select_gpus(
                profile,
                p.budgets,
                engine,
                context,
                util,
                p.notes,
                objective,
                cfg.planner.exclude_gpus,
                network_hop_s=hop,
            )
        from astra.planner import engines

        spec = (
            engines.llamacpp(result, "<model.gguf>")
            if engine == "llamacpp"
            else engines.vllm(result)
        )
        data = result.to_dict()
        data["launch"] = {"command": spec.shell(posix=True), "notes": list(spec.notes)}
        with self._lock:
            self.active_plan = data
        return data


def _static(name: str) -> bytes:
    return resources.files("astra.ui").joinpath("static", name).read_bytes()


def make_handler(state: UiState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"astra-ui/{__version__}"

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise AstraError("request too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise AstraError("JSON object expected")
            return data

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, _static("index.html"), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(200, state.state())
            elif path == "/api/models":
                self._json(
                    200,
                    {
                        "models": [
                            {
                                "key": k,
                                "name": a.display,
                                "params_b": round(a.n_params / 1e9, 1),
                                "layers": a.n_layers,
                            }
                            for k, a in CATALOG.items()
                        ],
                        "quants": [
                            {"key": k, "engines": list(q.engines)} for k, q in QUANTS.items()
                        ],
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            try:
                body = self._body()
                if path == "/api/plan":
                    self._json(200, state.make_plan(body))
                elif path == "/api/chat":
                    self._chat(body)
                else:
                    self._json(404, {"error": "not found"})
            except (AstraError, ValueError) as exc:
                self._json(400, {"error": str(exc)})

        def _chat(self, body: dict[str, Any]) -> None:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise AstraError("messages must be a non-empty list")
            payload = json.dumps(
                {
                    "messages": messages[-40:],
                    "stream": True,
                    "max_tokens": min(int(body.get("max_tokens", 1024)), 8192),
                    "temperature": float(body.get("temperature", 0.7)),
                }
            ).encode()
            req = urllib.request.Request(
                f"{state.engine_url}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                upstream = urllib.request.urlopen(req, timeout=300)
            except (urllib.error.URLError, OSError) as exc:
                self._json(503, {"error": f"engine at {state.engine_url} is not reachable: {exc}"})
                return
            with upstream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for line in upstream:
                        self.wfile.write(line)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    log.debug("client closed the chat stream")

        def log_message(self, fmt: str, *args: object) -> None:
            log.debug("%s " + fmt, self.address_string(), *args)

    return Handler


def serve(state: UiState, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    log.info("ASTRA console on http://%s:%d/", host, port)
    print(f"ASTRA console: http://{host}:{port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
