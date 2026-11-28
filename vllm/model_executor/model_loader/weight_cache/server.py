# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unix socket server for a preload daemon's local clients."""

import contextlib
import fcntl
import os
import select
import signal
import socket
import time
from collections.abc import Callable
from pathlib import Path

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.ipc_backends.serial_utils import (
    recv_msg,
    send_msg,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    WeightCacheKey,
    WeightCacheUnavailableError,
    ensure_private_socket_dir,
    get_current_device_uuid,
    get_socket_path,
    verify_peer_is_owner,
)
from vllm.model_executor.model_loader.weight_cache.transport import SeedTransport
from vllm.model_executor.model_loader.weight_cache.utils import (
    exported_weight_bytes,
    format_daemon_role,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.seed import (
    WEIGHT_COPY_LEASE_TIMEOUT_S,
    WeightShardSeed,
)

logger = init_logger(__name__)


class WeightCacheServer:
    def __init__(self, shard: WeightShardSeed, transport: SeedTransport) -> None:
        self.shard = shard
        self.transport = transport
        self._memory_bytes = exported_weight_bytes(shard.state.storages)
        self._watchers: list[socket.socket] = []

    def serve_forever(
        self, socket_dir: str | None, ready_callback: Callable[[], None]
    ) -> None:
        def stop(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        shard = self.shard
        path = Path(
            get_socket_path(
                get_current_device_uuid(), socket_dir, is_draft=shard.cache_key.is_draft
            )
        )
        ensure_private_socket_dir(str(path.parent), strict_perms=socket_dir is None)
        with contextlib.ExitStack() as cleanup:
            # Only the lock holder may replace the GPU's socket or remove it on exit.
            lock_fd = os.open(f"{path}.lock", os.O_CREAT | os.O_RDWR, 0o600)
            cleanup.callback(os.close, lock_fd)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise WeightCacheUnavailableError(
                    f"Another weight cache daemon already owns {path}"
                ) from e
            path.unlink(missing_ok=True)
            listener = cleanup.enter_context(
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            )
            listener.bind(str(path))
            cleanup.callback(path.unlink, missing_ok=True)
            path.chmod(0o600)
            listener.listen()
            cleanup.callback(self._close_watchers)
            # Keep both sockets available for acknowledgements until copies finish.
            cleanup.callback(self._drain_copies, listener)
            logger.info(
                "Weight cache %s daemon rank %d serving on %s",
                format_daemon_role(shard.cache_key.is_draft),
                shard.global_rank,
                path,
            )
            ready_callback()
            while True:
                conn, _ = listener.accept()
                self.handle_connection(conn)

    def _drain_copies(self, listener: socket.socket) -> None:
        seed = self.shard
        seed.start_release()
        deadline = time.monotonic() + WEIGHT_COPY_LEASE_TIMEOUT_S
        listener.settimeout(0.5)
        while seed.active_copies and time.monotonic() < deadline:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            conn.settimeout(5.0)
            self.handle_connection(conn)
        seed.finish_release()

    def handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                verify_peer_is_owner(conn)
                request = recv_msg(conn)
                send_msg(conn, self._dispatch(conn, request))
            except (ConnectionError, EOFError):
                logger.warning("Client disconnected mid-request")
            except Exception as e:
                logger.exception("Error handling weight cache client; continuing")
                with contextlib.suppress(OSError):
                    send_msg(conn, {"status": "error", "message": str(e)})

    def _dispatch(self, conn: socket.socket, request: dict) -> dict:
        cmd = request.get("cmd")
        if cmd == "get_state":
            return self._get_state(request.get("cache_key"))
        if cmd == "get_memory":
            return {
                "status": "ok",
                "memory_bytes": self._memory_bytes,
                "gpu_uuid": get_current_device_uuid(),
            }
        if cmd == "seed":
            return self.transport.handle_request(request)
        if cmd == "watch":
            self._watch(conn)
            return {"status": "ok"}
        return {"status": "error", "message": f"Unknown command {cmd!r}"}

    def _get_state(self, client_key: object) -> dict:
        shard = self.shard
        if not isinstance(client_key, WeightCacheKey):
            return {"status": "error", "message": "Missing cache_key"}
        mismatched = shard.cache_key.mismatched_fields(client_key)
        if mismatched:
            logger.warning("WeightCacheKey mismatch on fields: %s", mismatched)
            return {"status": "mismatch", "fields": mismatched}
        state = shard.state
        storages = [shard.backend.export_storage(record) for record in state.storages]
        logger.info_once(
            "Weight cache %s daemon rank %d exported %d storages (+%d aliases)",
            format_daemon_role(shard.cache_key.is_draft),
            shard.global_rank,
            len(storages),
            len(state.aliases),
        )
        return {
            "status": "ok",
            "backend": shard.backend_name,
            "storages": storages,
            "manifest": state.manifest,
            "aliases": state.aliases,
            "attrs": state.attrs,
            "gpu_uuid": get_current_device_uuid(),
        }

    def _watch(self, conn: socket.socket) -> None:
        # Engines send nothing after watch, so readability signals a disconnect.
        alive = []
        for watcher in self._watchers:
            readable, _, _ = select.select([watcher], [], [], 0)
            if readable:
                watcher.close()
            else:
                alive.append(watcher)
        # handle_connection closes conn after the response.
        alive.append(conn.dup())
        self._watchers = alive

    def _close_watchers(self) -> None:
        for watcher in self._watchers:
            watcher.close()
        self._watchers.clear()
