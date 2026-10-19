# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connectors for the weight cache channels.

`ClientConnector` is the channel client, used both by engine workers (through
`IpcModelLoader`) and by mirror daemons seeding from a peer. It owns target
resolution, dialing, timeouts and response decoding.

`DaemonConnector` is the channel server. It serves a `WeightCacheService` on
the two planes with their distinct trust domains, and owns every transport
invariant so the service never touches a socket:

- local Unix socket: pickle, 0700 directory, 0600 socket, per-GPU flock,
  SO_PEERCRED owner verification, the full command set;
- remote TCP: JSON only, 64 KiB / 10 s request budget, HMAC token compared
  before any payload is trusted, and a command whitelist of read-only
  commands.
"""

import contextlib
import fcntl
import hmac
import os
import socket
import threading
import time
from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    CacheConfigMismatchError,
    WeightCacheKey,
    WeightCacheState,
    WeightCacheUnavailableError,
    connect_daemon,
    dataclass_to_json,
    ensure_private_socket_dir,
    get_current_device_uuid,
    get_socket_path,
    json_to_dataclass,
    recv_json,
    recv_msg,
    send_json,
    send_msg,
    verify_peer_is_owner,
)
from vllm.model_executor.model_loader.weight_cache.seed import RDMA_SEED_SOURCE
from vllm.model_executor.model_loader.weight_cache.service import WeightCacheService

logger = init_logger("vllm.model_executor.model_loader.weight_cache.connector")

_SEED_TIMEOUT_S = 300.0
# A remote peer is untrusted until it presents the token, so it gets only this
# long, and this many bytes, to send its request (a command and a token).
_REMOTE_REQUEST_TIMEOUT_S = 10.0
_MAX_REMOTE_REQUEST_SIZE = 64 << 10


class ClientConnector:
    """Client side of the weight cache channel.

    The target is a Unix socket path (same-host daemon; the pickle plane,
    required for CUDA IPC handles) or a ``(host, port)`` pair (remote seed
    peer; the JSON plane, token-authenticated).
    """

    def __init__(
        self,
        target: str | tuple[str, int],
        *,
        token: str | None = None,
        connect_timeout_s: float = 5.0,
        state_timeout_s: float = 300.0,
        strict_perms: bool = False,
    ):
        self.target = target
        self.token = token
        self.connect_timeout_s = connect_timeout_s
        self.state_timeout_s = state_timeout_s
        # The auto-derived per-user socket directory is locked to 0700 and
        # checked strictly; an operator-configured path is their trust
        # decision, so only ownership/symlink safety is enforced there.
        self.strict_perms = strict_perms

    @classmethod
    def for_local_daemon(
        cls,
        *,
        socket_path: str | None = None,
        socket_dir: str | None = None,
        is_draft: bool = False,
        connect_timeout_s: float = 5.0,
        state_timeout_s: float = 300.0,
    ) -> "ClientConnector":
        """Connect to the daemon serving this process's GPU."""
        strict_perms = socket_path is None and socket_dir is None
        if socket_path is None:
            socket_path = get_socket_path(
                get_current_device_uuid(), socket_dir, is_draft=is_draft
            )
        return cls(
            socket_path,
            connect_timeout_s=connect_timeout_s,
            state_timeout_s=state_timeout_s,
            strict_perms=strict_perms,
        )

    def get_state(self, cache_key: WeightCacheKey) -> WeightCacheState:
        """Fetch this rank's tensors, keyed by the engine's fingerprint."""
        response = self._request(
            {"cmd": "get_state", "cache_config": cache_key}, self.state_timeout_s
        )
        status = response.get("status")
        if status == "mismatch":
            raise CacheConfigMismatchError(
                f"WeightCacheKey mismatch on fields: {response.get('fields')}"
            )
        if status != "ok":
            raise WeightCacheUnavailableError(
                f"Weight cache daemon error: {response.get('message')}"
            )
        return WeightCacheState(
            entries=response["entries"],
            aliases=response.get("aliases", {}),
            attrs=response.get("attrs", {}),
            gpu_uuid=response.get("gpu_uuid"),
        )

    def get_memory(self) -> int:
        """Bytes this daemon holds in GPU memory (for the engine's budget)."""
        response = self._request({"cmd": "get_memory"}, self.connect_timeout_s)
        if response.get("status") != "ok":
            raise WeightCacheUnavailableError(
                "Weight cache daemon rejected the memory query: "
                f"{response.get('message')}"
            )
        return int(response.get("memory_bytes", 0))

    def fetch_manifest(self, backend: str) -> dict:
        """Ask a source daemon for this rank's manifest and transfer seed.

        The raw response is returned for the caller to interpret (seeding is
        fail-closed, so unlike the engine's queries there is no fallback
        mapping here).
        """
        response = self._request(
            {"cmd": "fetch_manifest", "seed_backend": backend}, _SEED_TIMEOUT_S
        )
        if not isinstance(response, dict):
            raise RuntimeError(
                f"Seed daemon returned a malformed response: {response!r}"
            )
        if isinstance(self.target, tuple) and response.get("status") == "ok":
            response["cache_config"] = json_to_dataclass(
                WeightCacheKey, response.get("cache_config")
            )
        return response

    def release(self) -> None:
        """Ask the daemon to drop its cache (copy mode). Best-effort."""
        try:
            self._request({"cmd": "release"}, self.connect_timeout_s)
        except (WeightCacheUnavailableError, ConnectionError, OSError):
            logger.warning("Failed to ask the weight cache daemon to release")

    def _request(self, payload: dict, timeout: float) -> dict:
        if isinstance(self.target, str):
            with connect_daemon(
                self.target, timeout, strict_perms=self.strict_perms
            ) as conn:
                send_msg(conn, payload)
                return recv_msg(conn)
        if self.token is not None:
            payload = {**payload, "token": self.token}
        with socket.create_connection(self.target, timeout=timeout) as sock:
            send_json(sock, payload)
            return recv_json(sock)


class DaemonConnector:
    """Serves a `WeightCacheService` on the local and remote planes."""

    def __init__(
        self,
        service: WeightCacheService,
        *,
        socket_dir: str | None = None,
        is_draft: bool = False,
        listen_addr: tuple[str, int] | None = None,
        seed_token: str | None = None,
    ):
        self.service = service
        self.socket_dir = socket_dir
        self.is_draft = is_draft
        self.listen_addr = listen_addr
        self.seed_token = seed_token
        self.local_commands: dict[str, Callable[[dict], dict]] = {
            "get_state": service.get_state,
            "get_memory": service.get_memory,
            "fetch_manifest": service.fetch_manifest,
            "release": service.release,
        }

    @property
    def socket_path(self) -> str:
        return get_socket_path(
            get_current_device_uuid(), self.socket_dir, is_draft=self.is_draft
        )

    def serve_forever(self, ready_callback: Callable[[], None] | None = None) -> None:
        """Serve requests until terminated.

        The socket is only bound once the model is fully cached, so clients
        get a connection error (and fall back to disk) until the daemon is
        ready.

        Args:
            ready_callback: Invoked once the socket is bound and listening,
                so the launcher can report overall readiness.

        """
        socket_path = self.socket_path
        ensure_private_socket_dir(
            os.path.dirname(socket_path), strict_perms=self.socket_dir is None
        )
        # Hold an exclusive per-GPU lock for the daemon's lifetime so a second
        # daemon cannot remove this daemon's live socket and hijack the path.
        lock_fd = self._acquire_gpu_lock(socket_path)
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        server.listen()
        remote_server = None
        if self.listen_addr is not None:
            family = socket.AF_INET6 if ":" in self.listen_addr[0] else socket.AF_INET
            remote_server = socket.create_server(self.listen_addr, family=family)
            threading.Thread(
                target=self._serve_remote,
                args=(remote_server,),
                name=f"vllm-weight-cache-seed-{self.service.global_rank}",
                daemon=True,
            ).start()
        logger.info(
            "Weight cache %s daemon rank %d serving on %s%s",
            self.service.role,
            self.service.global_rank,
            socket_path,
            f" and {self.listen_addr[0]}:{self.listen_addr[1]} (seed peers)"
            if self.listen_addr is not None
            else "",
        )
        if ready_callback is not None:
            ready_callback()
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    try:
                        verify_peer_is_owner(conn)
                        self._handle_local(conn)
                    except (ConnectionError, EOFError):
                        logger.warning("Client disconnected mid-request")
                    except Exception as e:
                        # Report the error back instead of just closing the
                        # socket, but don't let it take the daemon down.
                        logger.exception(
                            "Error handling weight cache client; continuing"
                        )
                        with contextlib.suppress(OSError):
                            send_msg(conn, {"status": "error", "message": str(e)})
        finally:
            server.close()
            if remote_server is not None:
                remote_server.close()
            with self.service.state_lock:
                for source in self.service._seed_sources.values():
                    source.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            os.close(lock_fd)

    def _handle_local(self, conn: socket.socket) -> None:
        request = recv_msg(conn)
        handler = self.local_commands.get(request.get("cmd"))
        if handler is None:
            send_msg(
                conn,
                {
                    "status": "error",
                    "message": f"Unknown command {request.get('cmd')!r}",
                },
            )
            return
        with self.service.state_lock:
            response = handler(request)
        send_msg(conn, response)

    # -- remote seed plane ---------------------------------------------------

    def _serve_remote(self, remote_server: socket.socket) -> None:
        """Serve seed peers on their own thread, one connection at a time.

        A peer is untrusted until it presents the token, so none may hold up
        the engines on the local socket: requests are read here under a
        deadline, and only an authenticated one takes the state lock.
        """
        torch.accelerator.set_device_index(self.service.device_index)
        while True:
            try:
                conn, _ = remote_server.accept()
            except OSError:
                return  # The listener was closed on shutdown.
            with conn:
                try:
                    self._handle_remote_connection(conn)
                except (ConnectionError, EOFError, TimeoutError):
                    logger.warning("Seed peer disconnected or stalled mid-request")
                except Exception as e:
                    logger.exception("Error handling seed peer; continuing")
                    with contextlib.suppress(OSError):
                        send_json(conn, {"status": "error", "message": str(e)})

    def _handle_remote_connection(self, conn: socket.socket) -> None:
        """Serve one request from a remote peer over the JSON control plane.

        The remote plane never unpickles: recv_msg would deserialize a
        peer's bytes before its token could be compared, which would make the
        listener an unauthenticated code-execution surface. JSON parsing is
        memory-safe and the size cap and deadline bound it, so the token is
        checked against a decoded request rather than a decoded object
        graph.
        """
        request = recv_json(
            conn,
            max_size=_MAX_REMOTE_REQUEST_SIZE,
            deadline=time.monotonic() + _REMOTE_REQUEST_TIMEOUT_S,
        )
        presented = request.get("token") if isinstance(request, dict) else None
        if (
            not self.seed_token
            or not isinstance(presented, str)
            or not hmac.compare_digest(presented, self.seed_token)
        ):
            send_json(conn, {"status": "error", "message": "Unauthorized seed request"})
            return
        # Read-only commands only; everything that mutates or exports GPU
        # state stays on the owner-verified local socket.
        if request.get("cmd") != "fetch_manifest":
            send_json(
                conn,
                {
                    "status": "error",
                    "message": f"Command {request.get('cmd')!r} is only served locally",
                },
            )
            return
        # Authenticated: a full manifest may take a while to stream.
        conn.settimeout(_SEED_TIMEOUT_S)
        with self.service.state_lock:
            response = self._fetch_manifest_remote(request)
        send_json(conn, response)

    def _fetch_manifest_remote(self, request: dict) -> dict:
        backend = request.get("seed_backend", RDMA_SEED_SOURCE)
        if not isinstance(backend, str):
            return {"status": "error", "message": "Malformed seed_backend"}
        try:
            source = self.service.seed_source(backend)
            if source.is_node_local:
                raise ValueError(
                    f"Seed backend {backend!r} copies through CUDA IPC, which "
                    "cannot cross hosts; mirror it over this daemon's Unix socket"
                )
            response = self.service.fetch_manifest(request)
        except Exception as error:
            return {"status": "error", "message": str(error)}
        if response.get("status") == "ok":
            response["cache_config"] = dataclass_to_json(response["cache_config"])
        return response

    def _acquire_gpu_lock(self, socket_path: str) -> int:
        """Take an exclusive lock guarding this GPU's socket path.

        The lock is advisory and released automatically when the daemon exits
        (or crashes), so a stale socket is only ever removed by whoever owns
        the lock. A running daemon holding it makes a second daemon fail fast
        instead of clobbering the live socket.
        """
        lock_fd = os.open(f"{socket_path}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(lock_fd)
            raise WeightCacheUnavailableError(
                f"Another weight cache daemon already owns {socket_path}"
            ) from e
        return lock_fd
