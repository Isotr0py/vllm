# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import signal
import subprocess
import time
from argparse import ArgumentParser
from functools import cached_property
from types import TracebackType
from typing import BinaryIO, Self

import httpx

from vllm.utils.network_utils import join_host_port


class ServerProcess:
    VLLM_RESET_CACHE_ENDPOINTS = [
        "/reset_prefix_cache",
        "/reset_mm_cache",
        "/reset_encoder_cache",
    ]

    def __init__(
        self,
        server_cmd: list[str],
        after_bench_cmd: list[str],
        *,
        show_stdout: bool,
        http_client: httpx.Client | None = None,
        server_address: str | None = None,
        ready_url: str | None = None,
    ) -> None:
        super().__init__()

        self.server_cmd = server_cmd
        self.after_bench_cmd = after_bench_cmd
        self.show_stdout = show_stdout
        self.http = http_client or httpx
        self._server_address = server_address
        self._ready_url = ready_url
        self._stopped = True

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self, *, log: BinaryIO | None = None, dev_mode: bool = True) -> None:
        stdout: BinaryIO | int | None = log
        if log is None and not self.show_stdout:
            stdout = subprocess.DEVNULL

        env = None
        if dev_mode:
            # Need `VLLM_SERVER_DEV_MODE=1` for `_reset_caches`
            env = os.environ | {"VLLM_SERVER_DEV_MODE": "1"}

        # Create new process for clean termination
        self.launch_time = time.monotonic()
        self._server_process = subprocess.Popen(
            self.server_cmd,
            start_new_session=True,
            stdout=stdout,
            stderr=subprocess.STDOUT if log is not None else None,
            env=env,
        )
        self._stopped = False

    def stop(self, *, grace_period: float | None = None) -> bool:
        """Return True if this call had to force shutdown."""
        if self._stopped:
            return False
        process = self._server_process
        forced = grace_period is None
        try:
            if grace_period is not None:
                process.terminate()
                process.wait(timeout=grace_period)
        except subprocess.TimeoutExpired:
            forced = True
        finally:
            # Workers can outlive the group leader.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            self._stopped = True
        return forced

    def run_subcommand(self, cmd: list[str]):
        return subprocess.run(
            cmd,
            stdout=None if self.show_stdout else subprocess.DEVNULL,
            check=True,
        )

    def after_bench(self) -> None:
        if not self.after_bench_cmd:
            self.reset_caches()
            return

        self.run_subcommand(self.after_bench_cmd)

    @cached_property
    def server_address(self) -> str:
        if self._server_address:
            return self._server_address
        parser = ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--host", default="localhost")
        parser.add_argument("-p", "--port", type=int, default=8000)
        args, _ = parser.parse_known_args(self.server_cmd)
        return "http://" + join_host_port(args.host, args.port)

    @property
    def ready_url(self) -> str:
        return self._ready_url or f"{self.server_address}/health"

    def is_server_ready(self, timeout: float | None = None) -> bool:
        try:
            with self.http.stream(
                "GET", self.ready_url, timeout=timeout, follow_redirects=True
            ) as response:
                return response.status_code == 200
        except httpx.RequestError:
            return False

    def check_alive(self) -> None:
        returncode = self._server_process.poll()
        if returncode is not None:
            raise RuntimeError(f"Server process exited with return code {returncode}")

    def wait_until_ready(
        self,
        timeout: float,
        *,
        poll_interval: float = 1.0,
        request_timeout: float | None = None,
    ) -> None:
        start_time = time.monotonic()
        while not self.is_server_ready(timeout=request_timeout):
            self.check_alive()
            if time.monotonic() - start_time > timeout:
                raise TimeoutError(
                    f"Server failed to become ready within {timeout} seconds."
                )
            time.sleep(poll_interval)

    def reset_caches(self) -> None:
        server_cmd = self.server_cmd

        # Use `.endswith()` to match `/bin/...`
        if server_cmd[0].endswith("vllm"):
            server_address = self.server_address
            print(f"Resetting caches at {server_address}")

            for endpoint in self.VLLM_RESET_CACHE_ENDPOINTS:
                res = self.http.post(
                    server_address + endpoint, timeout=None, follow_redirects=True
                )
                res.raise_for_status()
        elif server_cmd[0].endswith("infinity_emb"):
            if "--vector-disk-cache" in server_cmd:
                raise NotImplementedError(
                    "Infinity server uses caching but does not expose a method "
                    "to reset the cache"
                )
        else:
            raise NotImplementedError(
                f"No implementation of `reset_caches` for `{server_cmd[0]}` server. "
                "Please specify a custom command via `--after-bench-cmd`."
            )
