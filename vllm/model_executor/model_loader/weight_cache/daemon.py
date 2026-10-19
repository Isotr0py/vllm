# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight cache daemon for fast engine restarts.

One daemon process per GPU holds the post-quantized, TP-sharded weights of its
rank in GPU memory and serves CUDA IPC handles to vLLM engines over a Unix
domain socket. Restarting engines map the weights via zero-copy IPC instead of
reloading from disk.

Launch one daemon per TP rank with a single command:

    vllm preload \\
        --model /path/to/model --tensor-parallel-size 4

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 \\
        --load-format ipc_cache

Tensor, expert, data and pipeline parallelism are supported.

For multi-node tensor parallelism, run one launcher per node with a shared
rendezvous so the global TP group forms across nodes (CUDA IPC handles are
node-local, so each node serves only its local GPUs' shards). Reuse the same
``--nnodes``/``--node-rank``/``--master-addr`` flags you pass the engine, plus a
``--weight-cache-master-port`` distinct from the engine's ``--master-port``:

    # node 0 (8 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 16 \\
        --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 \\
        --weight-cache-master-port 29600
    # node 1 (8 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 16 \\
        --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 \\
        --weight-cache-master-port 29600

The global TP rank of local GPU ``i`` on node ``r`` is
``r * (tp_size // nnodes) + i``, matching vLLM's contiguous per-node rank
assignment, so each engine worker maps its shard from the daemon on its own
node. With pipeline parallelism each node additionally runs one daemon per PP
stage; the global rank is ``pp_rank * tp_size + tp_rank``.

For data parallelism (e.g. a TP1 x DP16 x EP decode fleet) run one launcher per
node with the engine's DP placement flags. Local GPU ``i`` serves DP rank
``start_rank + i // tp_size`` and TP rank ``i % tp_size``, and all
``dp_size * tp_size`` daemons form one world group on
``--data-parallel-address``/``--weight-cache-master-port`` so the expert
shards are laid out exactly as in the engine:

    # node r (4 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 1 --enable-expert-parallel \\
        --data-parallel-size 16 --data-parallel-size-local 4 \\
        --data-parallel-start-rank 4r --data-parallel-address 10.0.0.1 \\
        --weight-cache-master-port 29600

Data parallelism also combines with multi-node tensor parallelism: pass both
flag sets (``--nnodes``/``--node-rank``/``--master-addr`` and the DP flags with
a reachable ``--data-parallel-address``). Each node then serves a contiguous
block of the ``dp_size * tp_size`` global ranks, e.g. TP8 x DP2 on 4 nodes:

    # node r (4 local GPUs)
    vllm preload \\
        --model /path/to/model --tensor-parallel-size 8 --enable-expert-parallel \\
        --nnodes 4 --node-rank r --master-addr 10.0.0.1 \\
        --data-parallel-size 2 --data-parallel-address 10.0.0.1 \\
        --weight-cache-master-port 29600

With MTP, EAGLE or EAGLE3 speculative decoding the launcher additionally
starts a draft daemon group that caches the draft model. It uses its own cache
key, Unix sockets (``*_draft.sock``) and rendezvous port
(``--weight-cache-draft-master-port``, default ``--weight-cache-master-port +
1``), so each process serves exactly one model role. Other draft types are not
cached and keep loading from disk in the engine.

This module owns only the daemon's lifecycle: configuration and rank
resolution, loading (from disk or from a seed peer), warmup, and readiness.
The wire transports live in `connector.py`, the command handlers' domain
logic in `service.py`, and the framing in `protocol.py`.
"""

import gc
import multiprocessing
from pathlib import Path
from typing import Any

import torch
from pybase64 import b64decode

from vllm.config import (
    ParallelConfig,
    VllmConfig,
    replace,
    set_current_vllm_config,
)
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_cache.connector import (
    ClientConnector,
    DaemonConnector,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    WeightCacheKey,
    check_ipc_quant_support,
    parse_seed_address,
)
from vllm.model_executor.model_loader.weight_cache.seed import (
    PEER_IPC_SEED_SOURCE,
    RDMA_SEED_SOURCE,
)
from vllm.model_executor.model_loader.weight_cache.service import (
    MirrorState,
    WeightCacheService,
)
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_warmup_runner,
    format_daemon_role,
    is_draft_model_cacheable,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_root,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger("vllm.model_executor.model_loader.weight_cache.daemon")


def seed_mismatches(
    local: WeightCacheKey,
    source: WeightCacheKey,
    model: str,
    source_model: str | None,
) -> list[str]:
    """Fields on which a seed source's shard differs from this daemon's.

    A mirror on a host without the checkpoint's safetensors cannot hash
    them, so its fingerprint falls back to the model path. It then matches
    the source by that path, as an engine without the files matches its
    daemon, rather than refusing every source that could hash its files.
    """
    mismatched = local.mismatched_fields(source)
    if "checkpoint" in mismatched and local.checkpoint == model == source_model:
        mismatched.remove("checkpoint")
    return mismatched


class WeightCacheDaemon:
    """Per-GPU process that loads one TP/PP shard and serves CUDA IPC handles.

    The daemon owns the lifecycle; the serving state and command handlers
    live in `WeightCacheService` (`service.py`), and the sockets in
    `DaemonConnector` (`connector.py`).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        global_rank: int,
        local_rank: int,
        distributed_init_method: str,
        socket_dir: str | None = None,
        is_draft: bool = False,
        dp_rank: int = 0,
        pp_rank: int = 0,
        seed_addr: str | None = None,
        seed_backend: str = PEER_IPC_SEED_SOURCE,
        listen_addr: tuple[str, int] | None = None,
        seed_token: str | None = None,
        device_offset: int = 0,
    ):
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            raise ValueError("The weight cache daemon does not support EPLB.")
        self.tp_size = parallel_config.tensor_parallel_size
        self.pp_size = parallel_config.pipeline_parallel_size
        self.dp_size = parallel_config.data_parallel_size
        if self.dp_size > 1:
            # The engine sets these on each DP rank's workers; the MoE/EP
            # layers read them to place their expert shards.
            if parallel_config.nnodes > 1:
                world_size = self.dp_size * self.tp_size
                local_world = world_size // parallel_config.nnodes
                start_dp_rank = parallel_config.node_rank * local_world // self.tp_size
            else:
                start_dp_rank = parallel_config.data_parallel_rank
            parallel_config = replace(
                parallel_config,
                data_parallel_rank=dp_rank,
                data_parallel_rank_local=dp_rank - start_dp_rank,
            )
            vllm_config = replace(vllm_config, parallel_config=parallel_config)
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
        rank_in_dp = global_rank - dp_rank * self.pp_size * self.tp_size
        self.tp_rank = rank_in_dp % self.tp_size
        self.pp_rank = rank_in_dp // self.tp_size
        self.dp_rank = dp_rank
        self.world_size = self.dp_size * self.pp_size * self.tp_size
        self.local_rank = local_rank
        # A peer's CUDA IPC handle names its device by index, so a same-host
        # mirror must see the source's GPUs at the indices the source used.
        # It therefore makes every device visible and shifts its own ranks
        # past the source's block instead of remapping with
        # CUDA_VISIBLE_DEVICES.
        self.device_index = local_rank + device_offset
        self.distributed_init_method = distributed_init_method
        self.socket_dir = socket_dir
        self.is_draft = is_draft
        self.role = format_daemon_role(is_draft)
        self.seed_addr = seed_addr
        self.seed_backend = seed_backend
        self.listen_addr = listen_addr
        self.seed_token = seed_token
        # Fingerprint before loading: process_weights_after_loading may
        # mutate hf_config.quantization_config.
        self.cache_config = WeightCacheKey.from_model_config(
            self.model_config,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank,
            dp_size=self.dp_size,
            dp_rank=dp_rank,
            is_draft=is_draft,
            enable_expert_parallel=parallel_config.enable_expert_parallel,
            expert_placement_strategy=parallel_config.expert_placement_strategy,
        )
        self.service = WeightCacheService(
            role=self.role,
            global_rank=global_rank,
            device_index=self.device_index,
            vllm_config=self.vllm_config,
            model_config=self.model_config,
            cache_config=self.cache_config,
        )

    def warmup(self) -> None:
        """Run the FlashInfer autotune pass against the cached model."""
        vllm_config = self.vllm_config
        # Mirrors the gate in the engine's kernel warmup.
        if (
            vllm_config.kernel_config.enable_flashinfer_autotune is False
            or not has_flashinfer()
            or not current_platform.has_device_capability(90)
            or is_draft_model_cacheable(vllm_config.speculative_config)
        ):
            return
        try:
            current_platform.update_block_size_for_backend(vllm_config)
            runner = build_warmup_runner(
                vllm_config,
                self.local_rank,
                is_draft=self.is_draft,
                model_config=self.model_config,
            )
            model = self.service.model
            assert model is not None, "warmup ran before load_model"
            runner.load_model(model=model)
            # The V2 runner requires skip_attn when no KV cache exists
            flashinfer_autotune(runner, skip_attn=vllm_config.use_v2_model_runner)
            logger.info(
                "Weight cache %s daemon rank %d tuned FlashInfer; the tuned "
                "configs are in the on-disk autotune cache",
                self.role,
                self.global_rank,
            )
        finally:
            # Free the runner and the dummy-run activations; the daemon's
            # weights are untouched.
            gc.collect()
            torch.accelerator.empty_cache()

    def load_model(self) -> None:
        torch.accelerator.set_device_index(self.device_index)
        if self.seed_addr is not None:
            # A mirror needs neither a rendezvous nor a model: it receives
            # this rank's finished tensors from a peer that already holds
            # them, which is the whole point of seeding.
            self._load_from_seed()
        else:
            # Outside the config context so the daemon keeps its own
            # rendezvous instead of the engine's; model parallel groups
            # (TP/DP/EP) are then carved out of this world group like in the
            # engine.
            init_distributed_environment(
                world_size=self.world_size,
                rank=self.global_rank,
                distributed_init_method=self.distributed_init_method,
                local_rank=self.device_index,
                backend=current_platform.dist_backend,
            )
            with set_current_vllm_config(self.vllm_config):
                ensure_model_parallel_initialized(self.tp_size, self.pp_size)
                self.service.model = self.get_model()
        # Loading and post-processing leave freed transients in the caching
        # allocator; return them so engines sharing the GPU can use them.
        gc.collect()
        torch.accelerator.empty_cache()
        model = self.service.model
        mirror = self.service.mirror
        logger.info(
            "Weight cache %s daemon rank %d %s (%s GiB allocated, %s GiB reserved)",
            self.role,
            self.global_rank,
            "loaded model"
            if model is not None
            else f"mirrored {len(mirror.tensors)} tensors"
            if mirror is not None
            else "cached nothing",
            format_gib(torch.accelerator.memory_allocated()),
            format_gib(torch.accelerator.memory_reserved()),
        )

    def _load_from_seed(self) -> None:
        """Fill this rank from a peer daemon instead of the checkpoint."""
        assert self.seed_addr is not None
        source = self.service.seed_source(self.seed_backend)
        target = parse_seed_address(self.seed_addr)
        if not isinstance(target, str) and source.is_node_local:
            raise ValueError(
                f"Seed backend {self.seed_backend!r} copies through CUDA IPC, "
                "which cannot cross hosts; seed from a Unix socket path or "
                f"pass --weight-cache-seed-backend {RDMA_SEED_SOURCE}"
            )
        response = ClientConnector(target, token=self.seed_token).fetch_manifest(
            self.seed_backend
        )
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Seed daemon {self.seed_addr} refused the mirror request: "
                f"{response.get('message', response)}"
            )
        source_config = response.get("cache_config")
        if not isinstance(source_config, WeightCacheKey):
            raise RuntimeError("Seed daemon returned no compatible cache config")
        mismatched = seed_mismatches(
            self.cache_config,
            source_config,
            self.model_config.model,
            response.get("model"),
        )
        if mismatched:
            hint = (
                f"; no safetensors under {self.model_config.model} to fingerprint "
                "here, so the mirror needs the source's --model path or the "
                "checkpoint files"
                if mismatched == ["checkpoint"]
                and self.cache_config.checkpoint == self.model_config.model
                else ""
            )
            raise RuntimeError(
                f"Seed daemon cache differs on fields {mismatched}; refusing "
                f"to mirror a different weight shard{hint}"
            )
        manifest = response["manifest"]
        tensors = source.fill(
            manifest,
            response["seed"],
            torch.device(current_platform.device_type, self.device_index),
        )
        if set(tensors) != set(manifest["tensors"]):
            raise RuntimeError("Seed backend did not fill the complete tensor manifest")
        self.service.mirror = MirrorState(
            tensors=tensors,
            kinds={
                name: "param" if view["is_param"] else "buffer"
                for name, view in manifest["tensors"].items()
            },
            aliases=dict(response.get("aliases") or {}),
            attrs=dict(response.get("attrs") or {}),
        )
        self._adopt_seeded_autotune_cache(response.get("autotune_cache"))

    def _adopt_seeded_autotune_cache(self, payload: Any) -> None:
        """Land the source's tuned table where this rank's engines look.

        The relative location comes from the source but is validated before
        it touches the filesystem; a missing or malformed payload only costs
        the engines a full autotune pass.
        """
        if not isinstance(payload, dict):
            return
        rel, encoded = payload.get("rel"), payload.get("data")
        if not isinstance(rel, str) or not isinstance(encoded, str):
            return
        parts = Path(rel).parts
        if (
            len(parts) != 2
            or parts[0] in ("", ".", "..")
            or not parts[1].startswith("autotune_configs")
        ):
            logger.warning(
                "Seed daemon sent a malformed autotune cache location %r", rel
            )
            return
        try:
            data = b64decode(encoded, validate=True)
        except ValueError:
            logger.warning(
                "Seed daemon sent an unreadable FlashInfer autotune cache; "
                "engines will tune for themselves"
            )
            return
        target = resolve_flashinfer_autotune_root() / parts[0] / parts[1]
        write_flashinfer_autotune_cache(target, data)
        logger.info(
            "Weight cache %s daemon rank %d adopted the source's FlashInfer "
            "autotune cache (%d bytes)",
            self.role,
            self.global_rank,
            len(data),
        )

    def get_model(self) -> torch.nn.Module:
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
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = loader.create_model(vllm_config, model_config)
            check_ipc_quant_support(model)
            loader.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    def serve_forever(self, ready_callback=None) -> None:
        """Hand the serving state to the connector and serve until terminated.

        The socket is only bound once the model is fully cached, so clients
        get a connection error (and fall back to disk) until the daemon is
        ready.
        """
        connector = DaemonConnector(
            self.service,
            socket_dir=self.socket_dir,
            is_draft=self.is_draft,
            listen_addr=self.listen_addr,
            seed_token=self.seed_token,
        )
        connector.serve_forever(ready_callback)


def _run_daemon(
    global_rank: int,
    local_rank: int,
    vllm_config: VllmConfig,
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue: "multiprocessing.Queue[tuple[str, int]]",
    is_draft: bool = False,
    dp_rank: int = 0,
    pp_rank: int = 0,
    seed_addr: str | None = None,
    seed_backend: str = PEER_IPC_SEED_SOURCE,
    listen_addr: tuple[str, int] | None = None,
    seed_token: str | None = None,
    device_offset: int = 0,
) -> None:
    daemon = WeightCacheDaemon(
        vllm_config,
        global_rank,
        local_rank,
        distributed_init_method,
        socket_dir,
        is_draft,
        dp_rank,
        pp_rank,
        seed_addr,
        seed_backend,
        listen_addr,
        seed_token,
        device_offset,
    )
    daemon.load_model()
    if seed_addr is None:
        # A mirror never built a world group, so it cannot tune in place; it
        # adopts the source's on-disk autotune cache during seeding instead.
        daemon.warmup()
    daemon.serve_forever(
        ready_callback=lambda: ready_queue.put((daemon.role, daemon.global_rank))
    )


def plan_local_ranks(
    parallel_config: ParallelConfig,
) -> list[tuple[int, int, int, int]]:
    """``(local_rank, dp_rank, pp_rank, tp_rank)`` for every local daemon.

    Global ranks enumerate DP, PP, then TP: ``global = dp_rank * pp_size *
    tp_size + pp_rank * tp_size + tp_rank``.
    With ``--nnodes`` the engine hands each node a contiguous block of global
    ranks, so node ``r`` serves ``node_rank * local + i``; without it a
    launcher's block starts at ``--data-parallel-start-rank * tp_size``.
    """
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    world_size = parallel_config.data_parallel_size * pp_size * tp_size
    if parallel_config.nnodes > 1:
        local_world_size = world_size // parallel_config.nnodes
        base = parallel_config.node_rank * local_world_size
    else:
        local_world_size = parallel_config.data_parallel_size_local * pp_size * tp_size
        base = parallel_config.data_parallel_rank * pp_size * tp_size
    placements = []
    for local_rank in range(local_world_size):
        global_rank = base + local_rank
        rank_in_dp = global_rank % (pp_size * tp_size)
        dp_rank = global_rank // (pp_size * tp_size)
        pp_rank, tp_rank = divmod(rank_in_dp, tp_size)
        placements.append((local_rank, dp_rank, pp_rank, tp_rank))
    return placements


def get_draft_daemon_config(vllm_config: VllmConfig) -> VllmConfig | None:
    """VllmConfig for the draft daemon group"""
    if not is_draft_model_cacheable(vllm_config.speculative_config):
        return None
    assert vllm_config.speculative_config is not None
    return vllm_config.speculative_config.apply_draft_overrides(vllm_config)


def _reject_unsupported_parallelism(parallel_config: ParallelConfig) -> None:
    """Reject placements the daemon cannot map."""
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
        return
    if dp_size == 1:
        return
    end_rank = (
        parallel_config.data_parallel_rank + parallel_config.data_parallel_size_local
    )
    if end_rank > dp_size:
        raise ValueError(
            "--data-parallel-start-rank + --data-parallel-size-local "
            f"({end_rank}) exceeds --data-parallel-size ({dp_size})"
        )
