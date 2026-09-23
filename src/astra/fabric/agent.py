"""``astra agent``: one per machine. Advertises GPUs and serves them over RPC.

* UDP discovery responder: answers ``ASTRA-DISCOVER`` datagrams with the API port.
* HTTP API: ``GET /v1/node`` (node descriptor), ``GET /healthz``.
* With ``--rpc``: supervises one llama.cpp ``rpc-server`` per GPU, each pinned to
  its GPU by UUID (ADR-0004) and restarted if it exits.

Security: llama.cpp's rpc-server is unauthenticated and must only be reachable
from the trusted cluster network (see docs/06-deployment, "Fabric"). The agent
API is read-only.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from astra import __version__
from astra.fabric.protocol import DISCOVER_MESSAGE, PROTOCOL, NodeDescriptor
from astra.hardware.models import GpuInfo

log = logging.getLogger("astra.agent")


# llama.cpp renamed rpc-server to ggml-rpc-server (field test FT-02, build b11149).
RPC_SERVER_NAMES = ("rpc-server", "ggml-rpc-server")


def find_rpc_server(preferred: str) -> str | None:
    """Resolve the rpc-server executable: the configured name/path, else known names."""
    for candidate in (preferred, *RPC_SERVER_NAMES):
        found = shutil.which(candidate)
        if found:
            return found
    return None


@dataclass(frozen=True)
class RpcServerSpec:
    uuid: str
    port: int
    argv: tuple[str, ...]
    env: dict[str, str]


def rpc_server_specs(
    gpus: Sequence[GpuInfo],
    binary: str = "rpc-server",
    host: str = "0.0.0.0",
    port_base: int = 50052,
    cache: bool = True,
) -> list[RpcServerSpec]:
    """One rpc-server per GPU; each process sees exactly one GPU (by UUID)."""
    specs = []
    for i, gpu in enumerate(sorted(gpus, key=lambda g: g.bus_id)):
        port = port_base + i
        argv = [binary, "--host", host, "--port", str(port)]
        if cache:
            argv.append("--cache")  # keep weights on local disk: re-loads skip the network
        specs.append(
            RpcServerSpec(
                uuid=gpu.uuid,
                port=port,
                argv=tuple(argv),
                env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu.uuid},
            )
        )
    return specs


class RpcSupervisor:
    """Start the rpc-servers and restart any that exit, until stopped."""

    def __init__(self, specs: Sequence[RpcServerSpec], restart_delay_s: float = 5.0) -> None:
        self.specs = list(specs)
        self.restart_delay_s = restart_delay_s
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _spawn(self, spec: RpcServerSpec) -> None:
        log.info("starting rpc-server for %s on port %d", spec.uuid, spec.port)
        self._procs[spec.uuid] = subprocess.Popen(list(spec.argv), env={**os.environ, **spec.env})

    def start(self) -> None:
        for spec in self.specs:
            self._spawn(spec)
        self._thread = threading.Thread(target=self._watch, name="rpc-supervisor", daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stop.wait(self.restart_delay_s):
            for spec in self.specs:
                proc = self._procs.get(spec.uuid)
                if proc is not None and proc.poll() is not None:
                    log.warning(
                        "rpc-server for %s exited (%s); restarting", spec.uuid, proc.returncode
                    )
                    self._spawn(spec)

    def stop(self) -> None:
        self._stop.set()
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self._procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def make_handler(describe: Callable[[], NodeDescriptor]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"astra-agent/{__version__}"

        def do_GET(self) -> None:
            if self.path == "/v1/node":
                try:
                    body, status = json.dumps(describe().to_dict()).encode(), 200
                except Exception as exc:  # report, don't die: the node may be mid-reboot of a GPU
                    body, status = json.dumps({"error": str(exc)}).encode(), 503
                ctype = "application/json"
            elif self.path == "/healthz":
                body, status, ctype = b"ok\n", 200, "text/plain; charset=utf-8"
            else:
                body, status, ctype = b"see /v1/node\n", 404, "text/plain; charset=utf-8"
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            log.debug("%s " + fmt, self.address_string(), *args)

    return Handler


class DiscoveryResponder:
    """Answer discovery datagrams with this node's name and API port."""

    def __init__(self, node: str, api_port: int, bind: str, port: int) -> None:
        self.node, self.api_port = node, api_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="discovery", daemon=True)

    @property
    def port(self) -> int:
        return int(self.sock.getsockname()[1])

    def start(self) -> None:
        self._thread.start()

    def _serve(self) -> None:
        reply = json.dumps({"protocol": PROTOCOL, "node": self.node, "api_port": self.api_port})
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(512)
            except TimeoutError:
                continue
            except OSError:
                break
            if data.strip() == DISCOVER_MESSAGE:
                self.sock.sendto(reply.encode(), addr)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.sock.close()


def serve_agent(
    describe: Callable[[], NodeDescriptor],
    node: str,
    bind: str,
    api_port: int,
    discovery_port: int | None,
    supervisor: RpcSupervisor | None = None,
) -> None:
    server = ThreadingHTTPServer((bind, api_port), make_handler(describe))
    responder = DiscoveryResponder(node, api_port, bind, discovery_port) if discovery_port else None
    if supervisor:
        supervisor.start()
    if responder:
        responder.start()
    log.info("astra agent '%s' on http://%s:%d/v1/node", node, bind, api_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if responder:
            responder.stop()
        if supervisor:
            supervisor.stop()
