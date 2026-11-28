# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-to-daemon seeding over authenticated JSON/TCP or local pickle.

A `Seedable` publishes one kind of state to peers, such as a warmup cache
or a captured weight shard. The target names what it
wants by kind and key fields; the source resolves the key to a payload; the
target validates and installs it:

    request:  {"kind": "flashinfer_autotune", <key fields>..., "token": "..."}
    response: {"status": "ok", "payload": <kind-specific>}
              {"status": "miss"} | {"status": "error", "message": "..."}

Remote payloads are JSON values. Large binary state (weights) must not be
inlined: the payload then describes the transfer (e.g. a storage manifest), and
the caller drives the data plane itself.

The TCP connection is unencrypted beyond the shared token; keep the listener
on a trusted network.
"""

import contextlib
import hmac
import socket
import threading
import time
from collections.abc import Iterable
from typing import Any, NamedTuple, Protocol

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.ipc_backends.serial_utils import (
    recv_msg,
    send_msg,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    WeightCacheUnavailableError,
    recv_json,
    send_json,
    verify_socket_owner,
)

logger = init_logger(__name__)

# Requests carry a kind and small key fields; payloads flow in responses.
_MAX_REMOTE_REQUEST_SIZE = 64 << 10
_REQUEST_TIMEOUT_S = 30.0
# Whole-message deadline for unauthenticated peers: a per-read timeout
# would let a peer trickle bytes forever and stall the serial accept loop.
_REMOTE_REQUEST_TIMEOUT_S = 10.0
_AUTHENTICATED_TIMEOUT_S = 300.0


class Seedable(Protocol):
    """One kind of state a prepared daemon publishes to peers."""

    kind: str

    def resolve(self, request: dict[str, Any]) -> Any | None:
        """Source side: produce the payload named by `request`.

        Validate the wire key first — raise an error on any malformed field
        (`ValueError` for bad values, `KeyError` for missing ones), before
        touching local state. The transport turns it into an error response.
        Return None on a miss.
        """
        ...

    def validate_remote(self, request: dict[str, Any]) -> None:
        """Raise ValueError to refuse the request on the remote JSON plane."""
        return None


class SeedConfig(NamedTuple):
    """A daemon's D2D endpoints and shared token (Unix paths need none)."""

    listen: tuple[str, int] | None = None
    source: str | tuple[str, int] | None = None
    token: str | None = None


class SeedTransport:
    """The daemon's D2D seeding endpoint: serves and fetches seedables."""

    def __init__(self, seed_config: SeedConfig | None = None):
        seed_config = seed_config or SeedConfig()
        listen, source, token = seed_config
        remote_source = source is not None and not isinstance(source, str)
        if (listen is not None or remote_source) and not token:
            raise ValueError("A daemon-to-daemon listener/source requires a token")
        self._seedables: dict[str, Seedable] = {}
        self._seed_config = seed_config
        self._server: socket.socket | None = None

    def fetch(self, kind: str, *, allow_miss: bool = False, **params: Any) -> Any:
        """Fetch one artifact from our source. Optional caches may return a miss."""
        response = self.request({"kind": kind, **params})
        if response is None:
            raise RuntimeError(f"Could not fetch {kind} from {self.source}")
        if allow_miss and response.get("status") == "miss":
            return None
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Source daemon refused the {kind} seed request: {response}"
            )
        return response.get("payload")

    @property
    def source(self) -> str | tuple[str, int] | None:
        """The configured source daemon address, if any."""
        return self._seed_config.source

    @property
    def listens(self) -> bool:
        return self._seed_config.listen is not None

    def handle_request(self, request: Any, *, remote: bool = False) -> dict[str, Any]:
        """Source side: answer one request; remote=True applies validate_remote."""
        kind_name = request.get("kind") if isinstance(request, dict) else None
        seedable = (
            self._seedables.get(kind_name) if isinstance(kind_name, str) else None
        )
        if seedable is None:
            return {"status": "error", "message": "unsupported seed kind"}
        try:
            if remote:
                seedable.validate_remote(request)
            payload = seedable.resolve(request)
        except (OSError, KeyError, TypeError, ValueError) as e:
            return {"status": "error", "message": str(e)}
        if payload is None:
            return {"status": "miss"}
        return {"status": "ok", "payload": payload}

    def request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Send one request to the source daemon; None with a log line on failure."""
        source = self._seed_config.source
        if source is None:
            return None
        label = payload.get("kind", "payload")
        try:
            if isinstance(source, str):
                response = self._request_local(source, payload)
            else:
                with socket.create_connection(
                    source, timeout=_REQUEST_TIMEOUT_S
                ) as sock:
                    send_json(sock, {**payload, "token": self._seed_config.token})
                    sock.settimeout(_AUTHENTICATED_TIMEOUT_S)
                    response = recv_json(sock)
        except (OSError, ValueError, WeightCacheUnavailableError) as e:
            logger.warning(
                "Cannot fetch %s from %s over D2D; falling back locally: %s",
                label,
                source,
                e,
            )
            return None
        if not isinstance(response, dict):
            logger.warning(
                "Source %s sent a malformed response for %s; falling back locally",
                source,
                label,
            )
            return None
        return response

    def _request_local(self, path: str, payload: dict[str, Any]) -> Any:
        """Fetch over the source's owner-verified Unix socket (pickle)."""
        verify_socket_owner(path, strict_perms=False)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_REQUEST_TIMEOUT_S)
            sock.connect(path)
            # Authenticated by ownership: seed handlers may be slow.
            sock.settimeout(_AUTHENTICATED_TIMEOUT_S)
            send_msg(sock, {"cmd": "seed", **payload})
            return recv_msg(sock)

    def publish(self, seedables: Iterable[Seedable]) -> tuple[str, int] | None:
        """Publish the prepared artifacts for local requests and optional TCP peers."""
        if self._server is not None:
            return self._server.getsockname()[:2]
        self._seedables = {seed.kind: seed for seed in seedables}
        listen = self._seed_config.listen
        if listen is None:
            return None
        family = socket.AF_INET6 if ":" in listen[0] else socket.AF_INET
        self._server = socket.create_server(listen, family=family)
        threading.Thread(
            target=self._serve,
            args=(self._server,),
            name="vllm-weight-cache-d2d",
            daemon=True,
        ).start()
        host, port = self._server.getsockname()[:2]
        return host, port

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None

    def _serve(self, server: socket.socket) -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                try:
                    request = recv_json(
                        conn,
                        max_size=_MAX_REMOTE_REQUEST_SIZE,
                        deadline=time.monotonic() + _REMOTE_REQUEST_TIMEOUT_S,
                    )
                    presented = (
                        request.get("token") if isinstance(request, dict) else None
                    )
                    if (
                        not self._seed_config.token
                        or not isinstance(presented, str)
                        or not hmac.compare_digest(presented, self._seed_config.token)
                    ):
                        send_json(
                            conn,
                            {"status": "error", "message": "Unauthorized request"},
                        )
                        continue
                    # Authenticated handlers (e.g. weight manifests) may be slow.
                    conn.settimeout(_AUTHENTICATED_TIMEOUT_S)
                    send_json(conn, self.handle_request(request, remote=True))
                except (OSError, ValueError) as e:
                    with contextlib.suppress(OSError):
                        send_json(conn, {"status": "error", "message": str(e)})
                except Exception as e:
                    # A handler bug or a payload json.dumps cannot serialize
                    # must answer the peer and leave the listener serving.
                    logger.exception("Error handling seed peer; continuing")
                    with contextlib.suppress(OSError):
                        send_json(conn, {"status": "error", "message": str(e)})
