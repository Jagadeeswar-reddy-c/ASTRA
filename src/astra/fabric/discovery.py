"""Find agents (UDP broadcast or an explicit peer list) and fetch their descriptors."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

from astra.errors import AstraError, ParseError
from astra.fabric.protocol import DISCOVER_MESSAGE, PROTOCOL, NodeDescriptor


@dataclass(frozen=True)
class Peer:
    host: str
    api_port: int

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.api_port}"


def parse_peers(text: str, default_port: int) -> list[Peer]:
    """'10.0.0.5, pc2:9900' -> peers (port defaults to the agent port)."""
    peers = []
    for item in (p.strip() for p in text.split(",")):
        if not item:
            continue
        host, sep, port = item.rpartition(":")
        if sep and port.isdigit():
            peers.append(Peer(host, int(port)))
        else:
            peers.append(Peer(item, default_port))
    return peers


def discover(
    port: int, timeout_s: float = 1.5, targets: Sequence[str] = ("255.255.255.255",)
) -> list[Peer]:
    """Broadcast a discovery datagram and collect replies until the timeout."""
    found: dict[tuple[str, int], Peer] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.2)
        for target in targets:
            try:
                sock.sendto(DISCOVER_MESSAGE, (target, port))
            except OSError:
                continue  # e.g. no broadcast route on this interface
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data, (host, _) = sock.recvfrom(1024)
            except TimeoutError:
                continue
            try:
                reply = json.loads(data)
                if reply.get("protocol") == PROTOCOL:
                    peer = Peer(host, int(reply["api_port"]))
                    found[(peer.host, peer.api_port)] = peer
            except (ValueError, KeyError, TypeError):
                continue
    return list(found.values())


def fetch_descriptor(peer: Peer, timeout_s: float = 5.0) -> NodeDescriptor:
    try:
        with urllib.request.urlopen(f"{peer.api_url}/v1/node", timeout=timeout_s) as resp:
            return NodeDescriptor.from_dict(json.loads(resp.read()))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise AstraError(f"agent at {peer.api_url} unreachable: {exc}") from exc


def gather(
    peers: Sequence[Peer], timeout_s: float = 5.0
) -> tuple[list[tuple[Peer, NodeDescriptor]], list[str]]:
    """Fetch every peer's descriptor; unreachable or invalid peers become errors, not crashes."""
    nodes, errors = [], []
    for peer in peers:
        try:
            nodes.append((peer, fetch_descriptor(peer, timeout_s)))
        except (AstraError, ParseError) as exc:
            errors.append(str(exc))
    return nodes, errors
