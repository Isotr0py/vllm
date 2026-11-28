# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure server startup, optionally alongside a long-running sidecar."""

import argparse
import json
import shlex
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from vllm.benchmarks.sweep.server import ServerProcess
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_ports_list, join_host_port
from vllm.utils.system_utils import set_env_var

Iteration = dict[str, Any]


@contextmanager
def cold_startup() -> Iterator[None]:
    """Give one cold start and its children fresh compilation caches."""
    from torch._inductor.utils import fresh_cache

    with (
        tempfile.TemporaryDirectory(prefix="vllm_startup_bench_cold_") as cache,
        set_env_var("VLLM_CACHE_ROOT", cache),
        fresh_cache(),
    ):
        yield


def _processes(
    args: argparse.Namespace, stack: ExitStack, http: httpx.Client
) -> tuple[ServerProcess, ServerProcess | None]:
    serve = shlex.split(args.serve_cmd)
    sidecar = shlex.split(args.pre_serve_cmd) if args.pre_serve_cmd else None
    ready_url = args.pre_serve_ready_url
    address = None
    if sidecar and not ready_url:
        if "preload" not in sidecar:
            raise ValueError("Specify --pre-serve-ready-url for this sidecar command")
        from vllm.entrypoints.common.cli_args import make_arg_parser

        parser = FlexibleArgumentParser()
        make_arg_parser(parser.add_subparsers().add_parser("serve"))
        options = parser.parse_args(serve[serve.index("serve") :])
        address = "http://" + join_host_port(options.host or "localhost", options.port)
        port = next(p for p in get_open_ports_list(2) if p != options.port)
        socket_dir = stack.enter_context(tempfile.TemporaryDirectory())
        sidecar += [
            "--weight-cache-health-host",
            "127.0.0.1",
            "--weight-cache-health-port",
            str(port),
            "--weight-cache-socket-dir",
            socket_dir,
        ]
        ready_url = f"http://127.0.0.1:{port}/health"
        config = dict(options.model_loader_extra_config or {})
        config.setdefault("state_timeout_s", args.server_ready_timeout)
        config.update(socket_dir=socket_dir, socket_path=None, fallback=False)
        if options.load_format != "ipc_cache":
            serve += ["--load-format", "ipc_cache"]
        serve.extend(
            f"--model-loader-extra-config.{k}={json.dumps(v)}"
            for k, v in config.items()
        )

    def process(cmd, **kwargs):
        server = ServerProcess(cmd, [], show_stdout=True, http_client=http, **kwargs)
        if server.is_server_ready(timeout=1):
            raise RuntimeError(f"{server.ready_url} already answers")
        return server

    server = process(serve, server_address=address)
    companion = process(sidecar, ready_url=ready_url) if sidecar else None
    return server, companion


@contextmanager
def _launch(
    process: ServerProcess,
    rows: dict[str, list[Iteration]],
    phase: str,
    log_dir: Path | None,
) -> Iterator[Iteration]:
    samples = rows.setdefault(phase, [])
    row: Iteration = {}
    samples.append(row)
    path = log_dir / f"{phase}-{len(samples)}.log" if log_dir else None
    with path.open("wb") if path else nullcontext() as log:
        if path:
            row["server_log"] = str(path)
        process.start(log=log, dev_mode=False)
        try:
            yield row
        finally:
            process.stop(grace_period=30)


def _measure(
    process: ServerProcess,
    row: Iteration,
    origin: float,
    timeout: float,
    client: OpenAI | None = None,
) -> None:
    process.wait_until_ready(timeout, poll_interval=0.05, request_timeout=1)
    row["ready_time"] = time.monotonic() - origin
    if client is None:
        return

    row["model"] = client.models.list().data[0].id
    with client.completions.create(
        model=row["model"],
        prompt="The capital of France is",
        temperature=0,
        max_tokens=8,
        stream=True,
    ) as stream:
        for chunk in stream:
            if any(choice.text for choice in chunk.choices):
                row["first_token_time"] = time.monotonic() - origin
                return
    raise RuntimeError("Completion ended before the first non-empty text")


def measure_server_startup(args: argparse.Namespace) -> dict[str, list[Iteration]]:
    rows: dict[str, list[Iteration]] = {}
    timeout = args.server_ready_timeout
    cold = args.num_iters_cold
    log_dir = None
    if args.output_json:
        output = Path(args.output_json)
        log_dir = Path(
            tempfile.mkdtemp(prefix=f"{output.stem}-logs-", dir=output.parent)
        )

    with ExitStack() as stack:
        http = stack.enter_context(httpx.Client(trust_env=False))
        server, sidecar = _processes(args, stack, http)
        client = OpenAI(
            base_url=server.server_address + "/v1",
            api_key="benchmark",
            http_client=http,
            max_retries=0,
            timeout=300,
        )
        # On a failed probe, stop both processes before joining their probe threads.
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
        resident = stack.enter_context(ExitStack())
        if sidecar and (cold or args.num_iters_warmup or args.num_iters_warm):
            for _ in range(max(1, cold)):
                resident.close()
                if cold:
                    resident.enter_context(cold_startup())
                phase = "sidecar" if cold else "sidecar_setup"
                daemon = resident.enter_context(_launch(sidecar, rows, phase, log_dir))
                origin = sidecar.launch_time
                if cold and args.async_serve:
                    with _launch(server, rows, "concurrent", log_dir) as serve:
                        serve["serve_launch_offset"] = server.launch_time - origin
                        probes = (
                            pool.submit(_measure, sidecar, daemon, origin, timeout),
                            pool.submit(
                                _measure, server, serve, origin, timeout, client
                            ),
                        )
                        for probe in as_completed(probes):
                            probe.result()
                else:
                    _measure(sidecar, daemon, origin, timeout)

        for phase, count in (
            ("cold", 0 if sidecar else cold),
            ("warmup", args.num_iters_warmup),
            ("warm", args.num_iters_warm),
        ):
            for _ in range(count):
                with (
                    cold_startup() if phase == "cold" else nullcontext(),
                    _launch(server, rows, phase, log_dir) as row,
                ):
                    _measure(server, row, server.launch_time, timeout, client)
    return rows
