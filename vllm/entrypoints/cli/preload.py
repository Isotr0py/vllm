# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `preload` subcommand for the vLLM CLI.

Launches one weight cache daemon per GPU; each daemon holds the
post-quantized, TP-sharded weights of its rank in GPU memory and serves
GPU IPC handles to vLLM engines over a Unix domain socket, so restarting
engines can map the weights via zero-copy IPC instead of reloading from disk.
The VMM backend uses CUDA/HIP VMM allocations on Linux:

    vllm preload --model /path/to/model --tensor-parallel-size 4

Engines then load from the daemons with:

    vllm serve /path/to/model --tensor-parallel-size 4 --load-format ipc_cache

Tensor, expert and data parallelism are supported, including across nodes;
see vllm/model_executor/model_loader/weight_cache/daemon.py for the rank
layout and rendezvous details.

A second replica's daemons can instead pull their weights from a loaded
replica (`--weight-cache-seed`); see `docs/features/preload.md`.
"""

import argparse
import typing
from functools import partial

from vllm.entrypoints.cli.preload_launcher import (
    LaunchInfo,
    RankMain,
    maybe_launch_workers,
    run_inline,
    supervise_prepared,
)
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.common.startup import VLLM_SUBCMD_PARSER_EPILOG

if typing.TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
        SeedAddress,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


def _validate_seed_args(
    args: argparse.Namespace,
    *,
    world_size: int,
    num_groups: int,
    nnodes: int,
    local_world_size: int,
    addresses: list["SeedAddress"],
) -> None:
    """Validate the seed CLI options before any rank starts loading."""
    for flag, host, port in (
        ("--seed-listen", args.seed_listen_host, args.seed_listen_port),
        ("--seed-source", args.seed_source_host, args.seed_source_port),
    ):
        if (host is None) != (port is None):
            raise ValueError(f"{flag}-host and {flag}-port must be provided together")
    if args.weight_cache_seed is not None and args.seed_source_host is not None:
        raise ValueError(
            "--weight-cache-seed conflicts with --seed-source-host/--seed-source-port"
        )
    unix_seed = any(isinstance(address, str) for address in addresses)
    remote_seed = any(not isinstance(address, str) for address in addresses)
    if unix_seed and remote_seed:
        raise ValueError(
            "--weight-cache-seed cannot mix Unix socket paths and "
            "host:base_port addresses"
        )
    if unix_seed and len(addresses) != local_world_size:
        raise ValueError(
            "Unix --weight-cache-seed requires one target socket per local "
            f"GPU; expected {local_world_size}, got {len(addresses)}"
        )
    if remote_seed and len(addresses) != nnodes:
        raise ValueError(
            "Remote --weight-cache-seed requires one host:base_port per source "
            f"node in node_rank order; expected {nnodes}, got {len(addresses)}"
        )
    # A token only protects the TCP planes: the listener and a remote source.
    has_remote = (
        args.seed_listen_host is not None
        or args.seed_source_host is not None
        or remote_seed
    )
    if has_remote != bool(args.seed_token):
        raise ValueError(
            "Remote daemon-to-daemon seeding requires --seed-token (and a "
            "token without a remote endpoint has no effect)"
        )
    for base_port in (
        args.seed_listen_port,
        args.seed_source_port,
        *(address[1] for address in addresses if isinstance(address, tuple)),
    ):
        if base_port is None:
            continue
        max_port = base_port + num_groups * world_size - 1
        if not 1 <= base_port <= 65535 or max_port > 65535:
            raise ValueError(
                f"Daemon-to-daemon port range {base_port}-{max_port} exceeds 65535"
            )


class PreloadSubcommand(CLISubcommand):
    """The `preload` subcommand for the vLLM CLI."""

    name = "preload"

    def prepare(self, argv: list[str]) -> None:
        super().prepare(argv)
        maybe_launch_workers(argv)

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> None:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader.weight_cache.ipc_backends import (
            IPC_BACKENDS,
        )

        EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--weight-cache-backend",
            choices=IPC_BACKENDS,
            default="torch_ipc",
            help="Local weight-sharing backend (default: torch_ipc). "
            "vmm_ipc uses CUDA/HIP VMM allocations and POSIX file descriptors "
            "on Linux.",
        )
        parser.add_argument(
            "--weight-cache-socket-dir",
            type=str,
            default=None,
            help="Directory for the daemon Unix sockets (default: tempdir).",
        )
        parser.add_argument(
            "--weight-cache-master-port",
            type=int,
            default=None,
            help="Rendezvous port for the daemon's own TP group. Must differ "
            "from the engine's --master-port (the daemons may still be loading "
            "when the engine starts) and match across nodes. Required when "
            "--nnodes > 1; defaults to a free port for single-node.",
        )
        parser.add_argument(
            "--weight-cache-draft-master-port",
            type=int,
            default=None,
            help="Rendezvous port for the MTP draft daemon group. Defaults to "
            "--weight-cache-master-port + 1 for multi-node, or a free port "
            "for single-node.",
        )
        parser.add_argument(
            "--weight-cache-health-port",
            type=int,
            default=None,
            help="Port for the parent /health endpoint; disabled by default.",
        )
        parser.add_argument(
            "--weight-cache-health-host",
            type=str,
            default="0.0.0.0",
            help="Host for the parent /health endpoint (default: 0.0.0.0).",
        )
        parser.add_argument(
            "--seed-listen-host",
            type=str,
            default=None,
            help="Host to bind for authenticated daemon-to-daemon requests.",
        )
        parser.add_argument(
            "--seed-listen-port",
            type=int,
            default=None,
            help="Base port for daemon-to-daemon requests; each daemon binds "
            "base_port + group offset + its global rank.",
        )
        parser.add_argument(
            "--seed-source-host",
            type=str,
            default=None,
            help="Host of a source daemon replica to seed warmup caches from.",
        )
        parser.add_argument(
            "--seed-source-port",
            type=int,
            default=None,
            help="Base port of the source daemon replica; each daemon dials "
            "base_port + group offset + its global rank.",
        )
        parser.add_argument(
            "--seed-token",
            type=str,
            default=None,
            help="Shared secret for daemon-to-daemon requests. The JSON "
            "connection is unencrypted; use a trusted network.",
        )
        parser.add_argument(
            "--weight-cache-seed",
            type=str,
            default=None,
            help="Fill these daemons from an already-loaded replica instead "
            "of the checkpoint: each daemon still builds its model (config "
            "files required, safetensors not) and pulls the post-processed "
            "weights from its rank-matched source. For same-host CUDA IPC, "
            "pass one target Unix socket per local GPU (comma separated, in "
            "device order); draft daemons use the corresponding *_draft.sock "
            "paths. For cross-host Mooncake RDMA, pass one host:base_port per "
            "source node (comma separated, in node_rank order); each rank "
            "selects its node's address and dials base_port + group offset + "
            "its global rank.",
        )
        parser.add_argument(
            "--warmup-cache-stores",
            type=str,
            default=None,
            help="Comma-separated shared store backends mirroring the "
            "warmup caches (e.g. 'mooncake', connected via "
            "MOONCAKE_CONFIG_PATH). The local disk cache is always the "
            "first tier; these add shared tiers that reads fall back to "
            "and tuned tables are published to.",
        )

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        info, ranks = PreloadSubcommand.build_plan(args)
        if len(ranks) == 1:
            run_inline(info, ranks[0])
        else:
            supervise_prepared(info, ranks)

    @staticmethod
    def build_plan(args: argparse.Namespace) -> tuple[LaunchInfo, list[RankMain]]:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader.weight_cache.daemon import (
            _reject_unsupported_parallelism,
            _run_daemon,
            daemon_serves_model,
            get_draft_daemon_config,
            plan_local_ranks,
        )
        from vllm.model_executor.model_loader.weight_cache.protocol import (
            check_ipc_platform_support,
        )
        from vllm.model_executor.model_loader.weight_cache.transport import SeedConfig
        from vllm.model_executor.model_loader.weight_cache.weight_seed.seed import (
            WEIGHT_COPY_LEASE_TIMEOUT_S,
        )
        from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
            mirror_device_indices,
            parse_seed_addresses,
            peer_cache_config,
            seed_address,
        )
        from vllm.usage.usage_lib import UsageContext
        from vllm.utils.network_utils import get_distributed_init_method, get_open_port

        engine_args = EngineArgs.from_cli_args(args)
        # Warmup context for flashinfer autotune.
        usage_context = (
            UsageContext.OPENAI_API_SERVER
            if engine_args.kernel_config.enable_flashinfer_autotune is not False
            else None
        )
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        load_format = vllm_config.load_config.load_format
        if load_format in ("ipc_cache", "peer_cache"):
            raise ValueError(
                f"--load-format {load_format} is not accepted here: the "
                "weight cache daemon loads from disk by default, and "
                "--weight-cache-seed switches it to peer_cache internally"
            )
        # Config-only so it can fail before any model loading; the quant
        # method check needs the created model and runs in
        # WeightCacheDaemon.get_model.
        check_ipc_platform_support()
        parallel_config = vllm_config.parallel_config
        _reject_unsupported_parallelism(parallel_config)
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        dp_size = parallel_config.data_parallel_size
        world_size = dp_size * pp_size * tp_size

        placements = plan_local_ranks(parallel_config)
        local_world_size = len(placements)
        if dp_size == 1:
            nnodes = parallel_config.nnodes
            node_rank = parallel_config.node_rank
            master_addr = parallel_config.master_addr
        else:
            master_addr = parallel_config.data_parallel_master_ip
            if parallel_config.nnodes > 1:
                nnodes = parallel_config.nnodes
                node_rank = parallel_config.node_rank
            else:
                nnodes = dp_size // parallel_config.data_parallel_size_local
                node_rank = parallel_config.data_parallel_rank // (
                    parallel_config.data_parallel_size_local
                )
            if nnodes > 1 and master_addr in ("127.0.0.1", "localhost"):
                raise ValueError(
                    "Data parallelism across nodes requires a reachable "
                    "--data-parallel-address for the daemon rendezvous"
                )

        # The daemon forms its own world group and holds it open while
        # serving, so it needs a rendezvous port distinct from the engine's.
        # All nodes must agree on it; single-node can auto-pick a free port.
        # The master address is the engine's --master-addr (TP across nodes)
        # or --data-parallel-address.
        if nnodes > 1 and args.weight_cache_master_port is None:
            raise ValueError(
                "--weight-cache-master-port is required when the daemons span nodes"
            )
        master_port = args.weight_cache_master_port or get_open_port()
        distributed_init_method = get_distributed_init_method(master_addr, master_port)

        # (is_draft, vllm_config, rendezvous) per daemon group.
        groups: list[tuple[bool, VllmConfig, str]] = [
            (False, vllm_config, distributed_init_method)
        ]
        draft_vllm_config = get_draft_daemon_config(vllm_config)
        if draft_vllm_config is not None:
            draft_master_port = args.weight_cache_draft_master_port or (
                master_port + 1 if nnodes > 1 else get_open_port()
            )
            if draft_master_port == master_port:
                raise ValueError(
                    "--weight-cache-draft-master-port must differ from "
                    "--weight-cache-master-port"
                )
            groups.append(
                (
                    True,
                    draft_vllm_config,
                    get_distributed_init_method(master_addr, draft_master_port),
                )
            )

        seed_addresses = parse_seed_addresses(args.weight_cache_seed)
        _validate_seed_args(
            args,
            world_size=world_size,
            num_groups=len(groups),
            nnodes=nnodes,
            local_world_size=local_world_size,
            addresses=seed_addresses,
        )
        warmup_cache_stores = [
            name.strip()
            for name in (args.warmup_cache_stores or "").split(",")
            if name.strip()
        ]

        def _seed_endpoint(
            host: str | None, base_port: int | None, offset: int
        ) -> tuple[str, int] | None:
            # Validated above: host and port are provided together.
            if host is None or base_port is None:
                return None
            return host, base_port + offset

        mirror_devices = None
        if seed_addresses and all(
            isinstance(address, str) for address in seed_addresses
        ):
            # Same-host mirror: place every local daemon on a GPU the
            # source replica does not use, inferred from the source's GPUs.
            mirror_devices = mirror_device_indices(seed_addresses, local_world_size)

        ranks: list[RankMain] = []
        serving = []
        for group_index, (is_draft, config, init_method) in enumerate(groups):
            # Draft daemons live past the target group's port block at every
            # source node.
            port_offset = group_index * world_size
            for local_rank, dp_rank, pp_rank, tp_rank in placements:
                global_rank = dp_rank * pp_size * tp_size + pp_rank * tp_size + tp_rank
                offset = port_offset + global_rank
                peer_source = seed_address(
                    seed_addresses,
                    local_rank,
                    global_rank,
                    port_offset,
                    node_rank=node_rank,
                    is_draft=is_draft,
                )
                source: str | tuple[str, int] | None
                if peer_source is not None:
                    daemon_config = peer_cache_config(config, is_draft=is_draft)
                    # The mirror also adopts the source's warmup caches (the
                    # FlashInfer autotune table) from the same address
                    # through the ordinary warmup-cache seeding path.
                    source = peer_source
                else:
                    daemon_config = config
                    source = _seed_endpoint(
                        args.seed_source_host, args.seed_source_port, offset
                    )
                seed_config = SeedConfig(
                    listen=_seed_endpoint(
                        args.seed_listen_host, args.seed_listen_port, offset
                    ),
                    source=source,
                    token=args.seed_token,
                )
                ranks.append(
                    partial(
                        _run_daemon,
                        global_rank=global_rank,
                        local_rank=local_rank,
                        vllm_config=daemon_config,
                        distributed_init_method=init_method,
                        socket_dir=args.weight_cache_socket_dir,
                        is_draft=is_draft,
                        dp_rank=dp_rank,
                        pp_rank=pp_rank,
                        seed_config=seed_config,
                        device_index=mirror_devices[local_rank]
                        if mirror_devices
                        else None,
                        warmup_cache_stores=warmup_cache_stores,
                        weight_cache_backend=args.weight_cache_backend,
                    )
                )
                serving.append(daemon_serves_model(is_draft, pp_rank, pp_size))
        return LaunchInfo(
            node_rank,
            nnodes,
            args.weight_cache_socket_dir,
            args.weight_cache_health_host,
            args.weight_cache_health_port,
            tuple(serving),
            WEIGHT_COPY_LEASE_TIMEOUT_S + 30,
        ), ranks

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        preload_parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description="Launch weight cache daemons (one per GPU) that "
            "keep post-quantized weights in GPU memory and serve them to "
            "engines over GPU IPC.",
            usage="vllm preload --model <model> [options]",
        )
        self.add_cli_args(preload_parser)
        preload_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return preload_parser


def cmd_init() -> list[CLISubcommand]:
    return [PreloadSubcommand()]
