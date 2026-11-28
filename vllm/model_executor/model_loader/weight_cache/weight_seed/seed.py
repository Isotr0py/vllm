# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish one prepared weight shard and track its outstanding peer copies."""

import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.transport import Seedable
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    SeedBackend,
    WeightMover,
    get_mover,
    parse_seed_backend,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    WeightsSeedPayload,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.ipc_backends.base import (
        IPCBackendBase,
    )
    from vllm.model_executor.model_loader.weight_cache.protocol import (
        WeightCacheKey,
        WeightCacheState,
    )


logger = init_logger(__name__)

WEIGHT_COPY_LEASE_TIMEOUT_S = 300.0


@dataclass(eq=False, kw_only=True)
class WeightShardSeed(Seedable):
    """A resident shard shared by engine attachment and peer seeding."""

    kind = "weights"
    state: "WeightCacheState"
    cache_key: "WeightCacheKey"
    model_name: str
    backend: "IPCBackendBase"
    backend_name: str
    device_index: int
    global_rank: int

    def __post_init__(self) -> None:
        self._movers: dict[SeedBackend, WeightMover] = {}
        self._active_copies = 0
        self._lease_lock = threading.Lock()
        self._release_pending = False

    @property
    def active_copies(self) -> int:
        with self._lease_lock:
            return self._active_copies

    def _mover(self, backend: SeedBackend) -> WeightMover:
        mover = self._movers.get(backend)
        if mover is None:
            mover = get_mover(
                backend,
                device_index=self.device_index,
                ipc_backend=self.backend,
            )
            self._movers[backend] = mover
        return mover

    def start_release(self) -> None:
        with self._lease_lock:
            self._release_pending = True
            if self._active_copies:
                logger.info(
                    "Weight cache daemon rank %d waiting for %d "
                    "weight-cache copy acknowledgement(s)",
                    self.global_rank,
                    self._active_copies,
                )

    def finish_release(self) -> None:
        with self._lease_lock:
            if self._active_copies:
                raise TimeoutError(
                    f"Timed out waiting for {self._active_copies} "
                    "weight-cache copy acknowledgement(s); cached "
                    "weights and NIC registrations were retained"
                )
            # A failed release stays pending: some regions may have been
            # unregistered, so publishing the old bookkeeping is unsafe.
            for mover in self._movers.values():
                mover.release()

    def _acknowledge(self) -> dict[str, bool]:
        with self._lease_lock:
            if not self._active_copies:
                raise ValueError("Unexpected weight-cache acknowledgement")
            self._active_copies -= 1
        return {"acknowledged": True}

    def validate_remote(self, request: dict[str, Any]) -> None:
        """Refuse node-local movers on the remote control plane."""
        if request.get("acknowledge"):
            return
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        if backend == PEER_IPC_BACKEND:
            raise ValueError(
                f"Seed backend {backend!r} requires node-local weight sharing; "
                "mirror it over this daemon's Unix socket"
            )

    def resolve(self, request: dict[str, Any]) -> WeightsSeedPayload | dict[str, bool]:
        """Describe this rank so a peer daemon can mirror it."""
        if request.get("acknowledge"):
            return self._acknowledge()
        backend = parse_seed_backend(request.get("seed_backend", PEER_IPC_BACKEND))
        with self._lease_lock:
            if self._release_pending:
                raise ValueError("Weights are being released")
            mover = self._mover(backend)
            state = self.state
            seed = mover.prepare_seed(state.storages)
            self._active_copies += 1
            return {
                "cache_config": asdict(self.cache_key),
                # Matched instead of the checkpoint fingerprint by a mirror
                # without the safetensors to hash (see seed_mismatches).
                "model": self.model_name,
                "manifest": state.manifest,
                "aliases": state.aliases,
                "attrs": state.attrs,
                "seed": seed,
            }
