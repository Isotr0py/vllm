# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import socket
import subprocess
from pathlib import Path

import pytest

from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.benchmark


def _run_startup(tmp_path: Path, *args: str, warm: int = 1):
    output = tmp_path / "startup.json"
    subprocess.run(
        [
            "vllm",
            "bench",
            "startup",
            *args,
            "--num-iters-cold=1",
            "--num-iters-warmup=0",
            f"--num-iters-warm={warm}",
            "--output-json",
            str(output),
        ],
        check=True,
        timeout=900,
    )
    return json.loads(output.read_text())


def test_bench_startup(tmp_path):
    results = _run_startup(tmp_path)
    for phase in ("cold", "warm"):
        (startup,) = results[f"{phase}_startup_times"]
        (compilation,) = results[f"{phase}_compilation_times"]
        assert startup > 0
        assert compilation >= 0
        assert results[f"avg_{phase}_startup_time"] == startup


def test_bench_startup_times_first_token_after_restart(tmp_path):
    port = get_open_port()
    results = _run_startup(
        tmp_path,
        "--serve-cmd",
        f"vllm serve facebook/opt-125m --enforce-eager --port {port}",
    )
    for phase in ("cold", "warm"):
        (ready,) = results[f"{phase}_ready_times"]
        (first_token,) = results[f"{phase}_first_token_times"]
        assert 0 < ready < first_token
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


@pytest.mark.parametrize(
    "launch_args", [(), ("--async-serve",)], ids=["wait", "concurrent"]
)
def test_bench_startup_reuses_one_preload_daemon(tmp_path, monkeypatch, launch_args):
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    model = "Qwen/Qwen3-0.6B"
    options = "--enforce-eager --max-model-len 128 --max-num-seqs 4"
    port = get_open_port()
    results = _run_startup(
        tmp_path,
        *launch_args,
        "--pre-serve-cmd",
        f"vllm preload --model {model} {options}",
        "--serve-cmd",
        f"vllm serve {model} {options} --port {port} --kv-cache-memory-bytes 32M",
        warm=2,
    )
    (daemon_ready,) = results["sidecar_ready_times"]
    assert daemon_ready > 0
    concurrent = results.get("concurrent_iterations", [])
    if launch_args:
        (pair,) = concurrent
        assert 0 <= pair["serve_launch_offset"] < daemon_ready
    else:
        assert not concurrent
    warm = results["warm_iterations"]
    assert len(warm) == 2
    for row in [*concurrent, *warm]:
        assert 0 < row["ready_time"] < row["first_token_time"]
        assert (
            "from the weight cache daemon via zero-copy sharing"
            in Path(row["server_log"]).read_text()
        )
