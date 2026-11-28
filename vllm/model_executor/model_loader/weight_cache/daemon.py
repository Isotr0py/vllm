# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare one rank's weights for fast engine restarts.

All ranks join distributed startup. Serving ranks load and warm up their shard,
capture its storages, then serve them after releasing the process groups.
See `docs/features/preload.md` for deployment examples.
"""

import contextlib
import gc
import multiprocessing
from collections.abc import Iterable

import torch

from vllm.config import (
    ParallelConfig,
    VllmConfig,
    replace,
    set_current_vllm_config,
)
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_world_group,
    init_distributed_environment,
    single_pp_stage,
)
from vllm.logger import configure_logging, init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_cache.protocol import (
    WeightCacheState,
    check_ipc_quant_support,
)
from vllm.model_executor.model_loader.weight_cache.server import WeightCacheServer
from vllm.model_executor.model_loader.weight_cache.transport import (
    SeedConfig,
    SeedTransport,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    DaemonRole,
    build_weight_cache_key,
    collect_tensors,
    export_model_attrs,
    format_daemon_role,
    is_draft_model_cacheable,
)
from vllm.model_executor.model_loader.weight_cache.warmup_cache.compiled import (
    CompileArtifactWarmupCache,
)
from vllm.model_executor.model_loader.weight_cache.warmup_cache.flashinfer_autotune import (  # noqa: E501
    FlashInferWarmupCache,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.loader import (
    PeerModelLoader,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.seed import (
    WeightShardSeed,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    build_manifest,
)
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger("vllm.model_executor.model_loader.weight_cache.daemon")


class WeightCacheDaemon:
    """Prepare a single rank for the resident weight cache server."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        global_rank: int,
        local_rank: int,
        *,
        is_draft: bool = False,
        device_index: int | None = None,
        warmup_cache_stores: Iterable[str] = (),
        weight_cache_backend: str = "torch_ipc",
    ):
        from vllm.model_executor.model_loader.weight_cache.ipc_backends import (
            IPC_BACKENDS,
        )

        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            raise ValueError("The weight cache daemon does not support EPLB.")
        ranks_per_dp = (
            parallel_config.pipeline_parallel_size
            * parallel_config.tensor_parallel_size
        )
        self.world_size = parallel_config.data_parallel_size * ranks_per_dp
        dp_rank, rank_in_dp = divmod(global_rank, ranks_per_dp)
        if parallel_config.data_parallel_size > 1:
            # The engine sets these on each DP rank's workers; the MoE/EP
            # layers read them to place their expert shards.
            start_dp_rank = (global_rank - local_rank) // ranks_per_dp
            parallel_config = replace(
                parallel_config,
                data_parallel_rank=dp_rank,
                data_parallel_rank_local=dp_rank - start_dp_rank,
            )
            vllm_config = replace(vllm_config, parallel_config=parallel_config)
        # Per-rank caches and expert placement follow the engine worker's rank.
        vllm_config.parallel_config.rank = rank_in_dp
        self.vllm_config = vllm_config
        # A draft daemon builds the draft with the target's VllmConfig, like
        # the engine does; only the ModelConfig differs.
        if is_draft:
            spec = vllm_config.speculative_config
            assert spec is not None and spec.draft_model_config is not None
            self.model_config = spec.draft_model_config
        else:
            self.model_config = vllm_config.model_config
        self.global_rank = global_rank
        # Same-host mirrors can place this rank on a different visible GPU.
        self.device_index = local_rank if device_index is None else device_index
        self.is_draft = is_draft
        self.role = format_daemon_role(is_draft)
        self.serves_model = daemon_serves_model(is_draft, global_rank, parallel_config)

        self.backend_name = weight_cache_backend
        self.backend = IPC_BACKENDS[weight_cache_backend]()
        self.model: torch.nn.Module | None = None
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config. Built by the shared helper
        # so the peer loader requests exactly the shard fingerprinted here.
        self.cache_key = build_weight_cache_key(
            vllm_config, self.model_config, is_draft=is_draft
        )

        warmup_cache_stores = tuple(warmup_cache_stores)
        self.flashinfer_cache = FlashInferWarmupCache(warmup_cache_stores)
        self.compile_cache = CompileArtifactWarmupCache(warmup_cache_stores)

    def prepare_weights(
        self, distributed_init_method: str, transport: SeedTransport
    ) -> WeightShardSeed | None:
        """Prepare and publish the local shard, then release startup resources."""
        shard = self._prepare_shard(distributed_init_method, transport)
        self.release_distributed()
        return shard

    def _prepare_shard(
        self, distributed_init_method: str, transport: SeedTransport
    ) -> WeightShardSeed | None:
        torch.accelerator.set_device_index(self.device_index)
        # Outside the config context so the daemon keeps its own rendezvous
        # instead of the engine's; model parallel groups (TP/DP/EP) are then
        # carved out of this world group like in the engine.
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.global_rank,
            distributed_init_method=distributed_init_method,
            local_rank=self.device_index,
            backend=current_platform.dist_backend,
        )
        if self.serves_model:
            self.compile_cache.adopt_before_load(transport)
        parallel = self.vllm_config.parallel_config
        with set_current_vllm_config(self.vllm_config):
            ensure_model_parallel_initialized(
                parallel.tensor_parallel_size, parallel.pipeline_parallel_size
            )
            if not self.serves_model:
                return None
            with single_pp_stage() if self.is_draft else contextlib.nullcontext():
                self.model = self.get_model(transport)
        # Loading and post-processing leave freed transients in the caching
        # allocator; return them so engines sharing the GPU can use them.
        gc.collect()
        torch.accelerator.empty_cache()
        logger.info(
            "Weight cache %s daemon rank %d loaded model (%s GiB allocated, "
            "%s GiB reserved)",
            self.role,
            self.global_rank,
            format_gib(torch.accelerator.memory_allocated()),
            format_gib(torch.accelerator.memory_reserved()),
        )
        self.flashinfer_cache.warmup(self.vllm_config, self.model, transport)
        if transport.listens:
            # Publish compiler outputs produced by FlashInfer warmup.
            self.compile_cache.prepare()
        shard = self.capture_state()
        # Peers may fetch only after weights and warmup caches are ready.
        transport.publish([shard, self.flashinfer_cache, self.compile_cache])
        return shard

    def capture_state(self) -> WeightShardSeed:
        """Materialize canonical post-warmup storages and drop the model."""
        assert self.model is not None
        tensors, kinds, aliases = collect_tensors(self.model)
        manifest, source_storages = build_manifest(tensors, kinds)
        attrs = export_model_attrs(self.model)
        from vllm.compilation.wrapper import remove_bytecode_hooks
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        remove_bytecode_hooks(self.model)
        self.model = None
        self.vllm_config.compilation_config.static_forward_context.clear()
        _ROPE_DICT.clear()
        del tensors, kinds
        gc.collect()
        torch.accelerator.empty_cache()
        storages = self.backend.materialize(source_storages)
        state = WeightCacheState(
            storages=storages,
            manifest=manifest,
            aliases=aliases,
            attrs=attrs,
        )

        return WeightShardSeed(
            state=state,
            cache_key=self.cache_key,
            model_name=self.model_config.model,
            backend=self.backend,
            backend_name=self.backend_name,
            device_index=self.device_index,
            global_rank=self.global_rank,
        )

    def release_distributed(self) -> None:
        """Release process groups and BLAS workspaces after loading and warmup.

        Serving only exports shared storage handles, so the communicators (NCCL,
        custom all-reduce and FlashInfer workspaces) and cached BLAS scratch
        would otherwise pin GPU memory for the daemon's lifetime.
        """
        # Import on the worker's device. Importing FlashInfer can initialize CUDA.
        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            destroy_fi_ar_workspace,
        )

        free_before, _ = torch.accelerator.get_memory_info()
        # No rank may tear down a communicator while a peer is still in a
        # warmup collective on it.
        get_world_group().barrier()
        # Created lazily by warmup's all-reduce fusion and owned by no group.
        destroy_fi_ar_workspace()
        destroy_model_parallel()
        destroy_distributed_environment()
        gc.collect()
        if current_platform.is_cuda():
            # ATen caches these separately from the model's exported storages.
            clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_workspaces is not None:
                clear_workspaces()
            else:
                logger.debug("PyTorch does not expose BLAS workspace cleanup")
        torch.accelerator.empty_cache()
        free_after, _ = torch.accelerator.get_memory_info()
        logger.info(
            "Weight cache %s daemon rank %d released startup resources (%s GiB freed)",
            self.role,
            self.global_rank,
            format_gib(free_after - free_before),
        )

    def get_model(self, transport: SeedTransport) -> torch.nn.Module:
        """Load the daemon's model, composed from the configured loader.

        Runs the quantization check after model creation but before the slow
        weight load, so an unsupported method fails fast. Online quantization
        always fails the check, so load_model's finalize step for it is
        unnecessary here.
        """
        vllm_config = self.vllm_config
        model_config = self.model_config
        load_config = vllm_config.load_config
        loader = get_model_loader(load_config)
        device_config = vllm_config.device_config
        target_device = torch.device(
            device_config.device if load_config.device is None else load_config.device
        )
        # Attention backends may take workspace at construction, like in the
        # engine's worker init.
        init_workspace_manager(target_device)
        if isinstance(loader, PeerModelLoader):
            return loader.load_model(
                vllm_config, model_config, cache_key=self.cache_key, transport=transport
            )
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = loader.create_model(vllm_config, model_config)
            check_ipc_quant_support(model)
            loader.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()


def _run_daemon(
    global_rank: int,
    local_rank: int,
    vllm_config: VllmConfig,
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue: "multiprocessing.Queue[tuple[DaemonRole, int]]",
    *,
    is_draft: bool = False,
    seed_config: SeedConfig | None = None,
    device_index: int | None = None,
    warmup_cache_stores: Iterable[str] = (),
    weight_cache_backend: str = "torch_ipc",
) -> None:
    # Spawned daemons start with logging unconfigured, like engine workers.
    if logging_config := getattr(vllm_config, "logging_config", None):
        configure_logging(logging_config)
    with contextlib.closing(SeedTransport(seed_config)) as transport:
        shard = WeightCacheDaemon(
            vllm_config,
            global_rank,
            local_rank,
            is_draft=is_draft,
            device_index=device_index,
            warmup_cache_stores=warmup_cache_stores,
            weight_cache_backend=weight_cache_backend,
        ).prepare_weights(distributed_init_method, transport)
        if shard is not None:
            ready = (format_daemon_role(is_draft), global_rank)
            WeightCacheServer(shard, transport).serve_forever(
                socket_dir, lambda: ready_queue.put(ready)
            )


def daemon_serves_model(
    is_draft: bool, global_rank: int, parallel_config: ParallelConfig
) -> bool:
    """Whether the daemon at this placement loads and serves a model.

    The engine runs the draft model only on the last pipeline stage. Draft
    daemons on other stages join the group's rendezvous and then exit.
    """
    pp_size = parallel_config.pipeline_parallel_size
    pp_rank = global_rank // parallel_config.tensor_parallel_size % pp_size
    return not is_draft or pp_rank == pp_size - 1


def get_draft_daemon_config(vllm_config: VllmConfig) -> VllmConfig | None:
    """VllmConfig for the draft daemon group"""
    if not is_draft_model_cacheable(vllm_config.speculative_config):
        return None
    assert vllm_config.speculative_config is not None
    return vllm_config.speculative_config.apply_draft_overrides(vllm_config)


def plan_local_ranks(parallel_config: ParallelConfig) -> range:
    """Contiguous global ranks on this node, ordered by DP, then PP, then TP."""
    dp_size = parallel_config.data_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    if parallel_config.nnodes > 1:
        world_size = dp_size * pp_size * tp_size
        if world_size % parallel_config.nnodes != 0:
            raise ValueError(
                f"--nnodes ({parallel_config.nnodes}) must evenly divide the "
                f"daemon world size ({world_size} = data-parallel-size "
                f"{dp_size} x pipeline-parallel-size {pp_size} x "
                f"tensor-parallel-size {tp_size})"
            )
        local_size = world_size // parallel_config.nnodes
        base = parallel_config.node_rank * local_size
        return range(base, base + local_size)

    first_dp = parallel_config.data_parallel_rank
    end_dp = first_dp + parallel_config.data_parallel_size_local
    if dp_size > 1 and end_dp > dp_size:
        raise ValueError(
            "--data-parallel-start-rank + --data-parallel-size-local "
            f"({end_dp}) exceeds --data-parallel-size ({dp_size})"
        )
    return range(first_dp * pp_size * tp_size, end_dp * pp_size * tp_size)
