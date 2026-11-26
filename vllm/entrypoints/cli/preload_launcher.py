# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process supervision for preload. Keep the multi-rank parent free of torch."""

import argparse
import contextlib
import logging
import multiprocessing as mp
import os
import pickle
import queue
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess

logger = logging.getLogger("vllm.entrypoints.cli.preload_launcher")
RankMain = Callable[[Callable[[], None]], None]


@dataclass
class LaunchInfo:
    node_rank: int
    nnodes: int
    socket_dir: str | None
    health_host: str
    health_port: int | None
    serving: tuple[bool, ...]
    shutdown_timeout: float


@contextlib.contextmanager
def _health(
    info: LaunchInfo, alive: Callable[[], bool], announce: Callable[..., None]
) -> Iterator[Callable[[], None]]:
    ready = threading.Event()

    def mark_ready() -> None:
        announce(
            "===== Weight cache daemon READY: node %d/%d serving %d "
            "daemon(s) in %s =====",
            info.node_rank,
            info.nnodes,
            sum(info.serving),
            info.socket_dir or "the default socket dir",
        )
        ready.set()

    if info.health_port is None:
        yield mark_ready
        return

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import Response

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.get("/health", include_in_schema=False)
    async def health() -> Response:
        return Response(status_code=200 if ready.is_set() and alive() else 503)

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=info.health_host,
            port=info.health_port,
            log_level="error",
            access_log=False,
        )
    )
    thread = threading.Thread(
        target=server.run, name="vllm-weight-cache-health", daemon=True
    )
    thread.start()
    try:
        yield mark_ready
    finally:
        ready.clear()
        server.should_exit = True
        thread.join()


def _handle_rank_signals() -> Callable[[], bool]:
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return lambda: running


def run_inline(info: LaunchInfo, run: RankMain) -> None:
    with _health(info, _handle_rank_signals(), logger.info) as ready:
        run(ready)


def _pack_plan(info: LaunchInfo, ranks: list[RankMain]) -> tuple[dict, list[bytes]]:
    # The supervisor can forward tasks without importing their GPU dependencies.
    return asdict(info), [pickle.dumps(rank) for rank in ranks]


def _rank(
    index: int, tasks: mp.Queue, events: mp.Queue, argv: list[str] | None
) -> None:
    if argv is not None:
        from vllm.entrypoints.cli.preload import PreloadSubcommand

        command = PreloadSubcommand()
        info, ranks = command.build_plan(command.parse_args(argv))
        run, *others = ranks
        events.put(_pack_plan(info, others))
        del ranks, others
    else:
        # Import while rank zero resolves the configuration, before receiving
        # this rank's task. The supervisor never imports the GPU runtime.
        from vllm.model_executor.model_loader.weight_cache import (
            daemon,  # noqa: F401
        )

        payload = tasks.get()
        if not payload:
            return
        run = pickle.loads(payload)

    if index == 0:
        # Keep accepting IPC clients while slower peers finish warming up.
        def announce_ready() -> None:
            logger.info(*tasks.get())
            events.put(index)

        threading.Thread(target=announce_ready, daemon=True).start()

    _handle_rank_signals()
    run(lambda: events.put(index))


def _supervise(
    width: int,
    argv: list[str] | None,
    prepared: tuple[dict, list[bytes]] | None = None,
) -> None:
    ctx = mp.get_context("spawn")
    workers: list[BaseProcess] = []
    inboxes: list[mp.Queue] = []
    events = ctx.Queue()
    info = None
    stop_signal = 0

    def stop(signum, frame):
        nonlocal stop_signal
        stop_signal = signum

    def check_workers() -> None:
        if stop_signal:
            raise SystemExit(128 + stop_signal)
        for i, proc in enumerate(workers):
            code = proc.exitcode
            if code is None:
                continue
            if code == 0 and info and (i >= len(info.serving) or not info.serving[i]):
                continue  # An unused slot or a draft rank on an earlier PP stage.
            logger.error(
                "Weight cache daemon rank %d exited (exitcode=%d); shutting down.",
                i,
                code,
            )
            raise SystemExit(128 - code if code < 0 else code or 1)

    def start_workers(count: int) -> None:
        while len(workers) < count:
            check_workers()
            i = len(workers)
            inbox = ctx.Queue()
            inboxes.append(inbox)
            proc = ctx.Process(
                target=_rank, args=(i, inbox, events, argv if i == 0 else None)
            )
            proc.start()
            workers.append(proc)

    def next_event():
        while True:
            check_workers()
            try:
                return events.get(timeout=0.25)
            except queue.Empty:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        start_workers(width)
        if prepared is None:
            prepared = next_event()
        layout, payloads = prepared
        info = LaunchInfo(**layout)
        del prepared
        start_workers(len(info.serving))
        first = 1 if argv is not None else 0
        for i in range(first, len(workers)):
            inboxes[i].put(payloads[i - first] if i < len(info.serving) else None)
        payloads.clear()
        serving = [p for p, serves in zip(workers, info.serving) if serves]

        def announce(*args):
            inboxes[0].put(args)
            next_event()  # Rank zero has logged READY with the user's logging config.

        with _health(
            info, lambda: all(p.exitcode is None for p in serving), announce
        ) as ready:
            pending = {i for i, serves in enumerate(info.serving) if serves}
            while pending:
                pending.remove(next_event())
            check_workers()
            ready()
            while True:
                check_workers()
                wait([p.sentinel for p in serving], timeout=1)
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        grace = info.shutdown_timeout if info and stop_signal else 10.0
        deadline = time.monotonic() + grace
        for proc in workers:
            proc.join(max(0.0, deadline - time.monotonic()))
        for proc in workers:
            if proc.is_alive():
                proc.kill()
            proc.join()
        for inbox in inboxes:
            inbox.cancel_join_thread()
            inbox.close()
        events.close()


def _exec_launcher(width: int, argv: list[str] | None, prepared=None) -> None:
    # Importing the vllm package itself imports torch. Re-exec this file by path
    # so the supervisor only loads its own dependencies. Preserve module lookup
    # for workers launched from an uninstalled source checkout.
    with tempfile.TemporaryFile() as plan:
        pickle.dump((sys.path, width, argv, prepared), plan)
        plan.seek(0)
        os.set_inheritable(plan.fileno(), True)
        os.execv(sys.executable, [sys.executable, __file__, str(plan.fileno())])


def supervise_prepared(info: LaunchInfo, ranks: list[RankMain]) -> None:
    _exec_launcher(0, None, _pack_plan(info, ranks))


def maybe_launch_workers(argv: list[str]) -> None:
    # An explicit local width is only a head start for imports. build_plan is
    # authoritative for placement, groups, validation and the number of workers.
    parser = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False, exit_on_error=False
    )
    parser.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    parser.add_argument("-pp", "--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("-dpl", "--data-parallel-size-local", type=int, default=1)
    parser.add_argument("-n", "--nnodes", type=int, default=1)
    try:
        # FlexibleArgumentParser accepts underscores in long option names.
        options = []
        for arg in argv:
            if arg.startswith("--"):
                name, sep, value = arg.partition("=")
                arg = name.replace("_", "-") + sep + value
            options.append(arg)
        hint, _ = parser.parse_known_args(options)
        width = (
            hint.tensor_parallel_size
            * hint.pipeline_parallel_size
            * hint.data_parallel_size_local
        )
        if (
            hint.nnodes == 1
            and width > 1
            and not any(a.startswith(("-h", "--help", "--config")) for a in argv)
        ):
            # Bound speculative workers before the full CLI validates the input.
            _exec_launcher(min(width, 8), argv)
            return
    except argparse.ArgumentError:
        pass  # Let the full parser report the invalid argument.


if __name__ == "__main__":
    with os.fdopen(int(sys.argv[1]), "rb") as plan:
        paths, *launch_args = pickle.load(plan)
    sys.path[:] = paths
    _supervise(*launch_args)
