# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domain logic of a weight cache daemon rank, decoupled from the wire.

`WeightCacheService` owns the serving state (the loaded model, or the mirror
received from a seed peer) and answers the daemon's commands as plain
``request dict -> response dict`` calls. It never touches a socket: the
transports live in `connector.py`, the framing in `protocol.py`, and the
process lifecycle (loading, warmup, readiness) in `daemon.py`.
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from pybase64 import b64encode

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    TensorEntry,
    WeightCacheKey,
    get_current_device_uuid,
)
from vllm.model_executor.model_loader.weight_cache.seed import (
    PEER_IPC_SEED_SOURCE,
    WeightCacheSeedSource,
    build_manifest,
    get_seed_source,
)
from vllm.model_executor.model_loader.weight_cache.utils import export_model_attrs
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    autotune_file_with_rank,
    resolve_flashinfer_autotune_dir,
    resolve_flashinfer_autotune_root,
)

logger = init_logger("vllm.model_executor.model_loader.weight_cache.service")

# The tuned table rides inside the seed response, so it shares that message's
# size budget; kilobytes in practice.
_MAX_SEED_AUTOTUNE_BYTES = 16 << 20


@dataclass
class MirrorState:
    """This rank's tensors as received from a peer daemon.

    A mirror deliberately has no model: it never reads the checkpoint, so it
    holds the finished tensors plus the metadata a client would otherwise
    recover from the model.
    """

    tensors: dict[str, torch.Tensor]
    kinds: dict[str, str]
    """Per-tensor "param" or "buffer"."""
    aliases: dict[str, str] = field(default_factory=dict)
    attrs: dict[str, bool] = field(default_factory=dict)


def collect_tensors(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, str]]:
    """Collect a model's tensors, preserving tied-parameter aliases.

    ``named_parameters``/``named_buffers`` are iterated with
    ``remove_duplicate=False`` so tied weights (e.g. ``lm_head.weight`` sharing
    storage with ``embed_tokens.weight``) are not silently dropped. Each unique
    tensor is kept once; every additional name that refers to the same tensor
    object is recorded in the returned alias map so the client can
    re-establish the shared identity instead of allocating uninitialized
    memory for it.

    Returns:
        A ``(tensors, kinds, aliases)`` triple: ``tensors`` maps a canonical
        name to its tensor, ``kinds`` maps it to "param" or "buffer", and
        ``aliases`` maps each duplicate name to its canonical name.

    """
    tensors: dict[str, torch.Tensor] = {}
    kinds: dict[str, str] = {}
    aliases: dict[str, str] = {}
    canonical_by_id: dict[int, str] = {}

    def _add(name: str, tensor: torch.Tensor, kind: str) -> None:
        canonical = canonical_by_id.get(id(tensor))
        if canonical is not None:
            aliases[name] = canonical
            return
        canonical_by_id[id(tensor)] = name
        tensors[name] = tensor
        kinds[name] = kind

    for name, param in model.named_parameters(remove_duplicate=False):
        _add(name, param, "param")
    # named_buffers includes non-persistent buffers (e.g. rotary embedding
    # caches) that state_dict would miss.
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in tensors or name in aliases:
            continue
        _add(name, buffer, "buffer")
    return tensors, kinds, aliases


def export_entries(
    model: torch.nn.Module,
) -> tuple[dict[str, TensorEntry], dict[str, str]]:
    """Export a model's tensors as ``TensorEntry``s, preserving aliases.

    CUDA reduction arguments must be exported separately for each consumer so
    that PyTorch registers a reference for each IPC mapping's lifetime.

    Returns:
        A ``(entries, aliases)`` pair, as `collect_tensors` but with each
        tensor exported.

    """
    tensors, kinds, aliases = collect_tensors(model)
    entries = {
        name: TensorEntry.from_tensor(tensor, kinds[name])
        for name, tensor in tensors.items()
    }
    return entries, aliases


class WeightCacheService:
    """Answers weight cache commands against one rank's serving state.

    All commands take and return plain dicts (the wire payload without
    framing). Request-level problems are reported as ``{"status": ...}``
    responses; anything that raises is a bug the transport reports back and
    survives. ``state_lock`` serializes requests across the local and remote
    planes; the transports acquire it, so the lock order (authenticate first,
    lock after) stays a transport property.
    """

    def __init__(
        self,
        *,
        role: str,
        global_rank: int,
        device_index: int,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        cache_config: WeightCacheKey,
        seed_sources: dict[str, WeightCacheSeedSource] | None = None,
    ):
        self.role = role
        self.global_rank = global_rank
        self.device_index = device_index
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.cache_config = cache_config
        # Shared with the daemon's load path, which fills it during seeding.
        self._seed_sources = seed_sources if seed_sources is not None else {}
        self.state_lock = threading.Lock()
        self.model: torch.nn.Module | None = None
        self.mirror: MirrorState | None = None

    @property
    def has_weights(self) -> bool:
        return self.model is not None or self.mirror is not None

    def seed_source(self, backend: str) -> WeightCacheSeedSource:
        source = self._seed_sources.get(backend)
        if source is None:
            source = get_seed_source(backend)
            self._seed_sources[backend] = source
        return source

    # -- commands ----------------------------------------------------------

    def get_state(self, request: dict) -> dict:
        client_config = request.get("cache_config")
        if not isinstance(client_config, WeightCacheKey):
            return {"status": "error", "message": "Missing cache_config"}
        mismatched = self.cache_config.mismatched_fields(client_config)
        if mismatched:
            logger.warning("WeightCacheKey mismatch on fields: %s", mismatched)
            return {"status": "mismatch", "fields": mismatched}
        if not self.has_weights:
            return {"status": "error", "message": "Weights were released"}
        entries, aliases, attrs = self._export_state()
        logger.info_once(
            "Weight cache %s daemon rank %d sent %d tensors (+%d aliases) to engine",
            self.role,
            self.global_rank,
            len(entries),
            len(aliases),
        )
        return {
            "status": "ok",
            "entries": entries,
            "aliases": aliases,
            "attrs": attrs,
            "gpu_uuid": get_current_device_uuid(),
        }

    def get_memory(self, request: dict) -> dict:
        return {
            "status": "ok",
            "memory_bytes": torch.accelerator.memory_allocated(),
        }

    def fetch_manifest(self, request: dict) -> dict:
        """Describe this rank so a peer can mirror it."""
        if not self.has_weights:
            return {"status": "error", "message": "Weights were released"}
        backend = request.get("seed_backend", PEER_IPC_SEED_SOURCE)
        try:
            tensors, kinds, aliases, attrs = self._seed_state()
            manifest, storages = build_manifest(tensors, kinds)
            seed = self.seed_source(backend).prepare_seed(
                storages, get_current_device_uuid()
            )
        except Exception as error:
            return {"status": "error", "message": str(error)}
        return {
            "status": "ok",
            "cache_config": self.cache_config,
            # Matched instead of the checkpoint fingerprint by a mirror
            # without the safetensors to hash (see seed_mismatches).
            "model": self.model_config.model,
            "manifest": manifest,
            "aliases": aliases,
            "attrs": attrs,
            "seed": seed,
            # Kilobytes next to the weights, and it saves the mirror's first
            # engine the autotune pass, so every plane carries it.
            "autotune_cache": self._read_autotune_cache(),
        }

    def release(self, request: dict) -> dict:
        # Unregister from the NIC before the memory goes back to the driver.
        for source in self._seed_sources.values():
            source.release()
        self.model = None
        self.mirror = None
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d released cached weights",
            self.role,
            self.global_rank,
        )
        return {"status": "ok"}

    # -- internals ----------------------------------------------------------

    def _export_state(self) -> tuple[dict[str, TensorEntry], dict[str, str], dict]:
        """Export this rank's tensors, aliases and flags for one client.

        Entries are built per request because PyTorch registers a reference
        for each IPC mapping's lifetime, so every client needs its own
        handles. A mirror carries the aliases and flags it was seeded with,
        since it has no model to recover them from.
        """
        if self.model is not None:
            entries, aliases = export_entries(self.model)
            return entries, aliases, export_model_attrs(self.model)
        mirror = self.mirror
        assert mirror is not None
        entries = {
            name: TensorEntry.from_tensor(tensor, mirror.kinds[name])
            for name, tensor in mirror.tensors.items()
        }
        return entries, dict(mirror.aliases), dict(mirror.attrs)

    def _seed_state(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, str], dict]:
        """This rank's canonical tensors, their kinds, aliases and flags."""
        if self.model is not None:
            tensors, kinds, aliases = collect_tensors(self.model)
            return tensors, kinds, aliases, export_model_attrs(self.model)
        mirror = self.mirror
        assert mirror is not None
        return mirror.tensors, mirror.kinds, dict(mirror.aliases), dict(mirror.attrs)

    def _autotune_cache_path(self) -> Path:
        """This rank's on-disk FlashInfer autotune cache file.

        The daemon's world group spans the same ranks the engine's does, so
        both derive the same per-rank file (see ``autotune_file_with_rank``).
        Resolved on this daemon's own config object, which warmup has
        finalized (e.g. hybrid block-size alignment) by the time seeds are
        served — the same finalization an engine applies before its lookup.
        """
        base = (
            resolve_flashinfer_autotune_dir(self.vllm_config) / "autotune_configs.json"
        )
        return autotune_file_with_rank(
            base,
            self.vllm_config.parallel_config.data_parallel_rank,
            self.global_rank,
        )

    def _read_autotune_cache(self) -> dict[str, str] | None:
        """This rank's tuned table and its cache-root-relative location.

        The location travels with the bytes because a mirror cannot recompute
        the cache directory: warmup finalizes the config the hash is keyed on
        and a mirror never warms up. ``None`` when this daemon never wrote a
        table (autotune disabled, or the single-GPU managed store, which has
        no v1 file to carry); the mirror's first engine then tunes, as
        without seeding.
        """
        try:
            path = self._autotune_cache_path()
            data = path.read_bytes() if path.is_file() else None
            rel = str(path.relative_to(resolve_flashinfer_autotune_root()))
        except (OSError, ValueError):
            logger.exception("Failed to read the FlashInfer autotune cache")
            return None
        if data is None:
            return None
        if len(data) > _MAX_SEED_AUTOTUNE_BYTES:
            logger.warning(
                "FlashInfer autotune cache is %d bytes, over the %d byte seed "
                "budget; the mirror's first engine will tune instead",
                len(data),
                _MAX_SEED_AUTOTUNE_BYTES,
            )
            return None
        return {
            "rel": rel,
            "data": b64encode(data).decode("ascii"),
        }
