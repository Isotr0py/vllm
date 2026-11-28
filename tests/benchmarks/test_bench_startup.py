# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from vllm.utils.network_utils import get_open_port, get_open_ports_list


@pytest.mark.benchmark
def test_bench_startup(tmp_path):
    output = tmp_path / "engine.json"
    subprocess.run(
        [
            "vllm",
            "bench",
            "startup",
            "--num-iters-cold",
            "1",
            "--num-iters-warmup",
            "0",
            "--num-iters-warm",
            "1",
            "--output-json",
            str(output),
        ],
        check=True,
        timeout=900,
    )
    results = json.loads(output.read_text())
    for phase in ("cold", "warm"):
        (startup,) = results[f"{phase}_startup_times"]
        (compilation,) = results[f"{phase}_compilation_times"]
        assert startup > 0
        assert compilation >= 0
        assert results[f"avg_{phase}_startup_time"] == startup


@pytest.mark.benchmark
def test_bench_startup_times_first_token_after_restart(tmp_path):
    port = get_open_port()
    output = tmp_path / "startup.json"
    cmd = [
        "vllm",
        "bench",
        "startup",
        "--serve-cmd",
        f"vllm serve facebook/opt-125m --enforce-eager --port {port}",
        "--num-iters-cold",
        "1",
        "--num-iters-warmup",
        "0",
        "--num-iters-warm",
        "1",
        "--output-json",
        str(output),
    ]
    subprocess.run(cmd, check=True, timeout=900)
    results = json.loads(output.read_text())
    for phase in ("cold", "warm"):
        (ready,) = results[f"{phase}_ready_times"]
        (first_token,) = results[f"{phase}_first_token_times"]
        assert 0 < ready < first_token
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


@pytest.mark.benchmark
@pytest.mark.parametrize("async_serve", [False, True], ids=["wait", "concurrent"])
def test_bench_startup_reuses_one_preload_daemon(tmp_path, monkeypatch, async_serve):
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    model = "Qwen/Qwen3-0.6B"
    options = "--enforce-eager --max-model-len 128 --max-num-seqs 4"
    output = tmp_path / "preload.json"
    subprocess.run(
        [
            "vllm",
            "bench",
            "startup",
            "--pre-serve-cmd",
            f"vllm preload --model {model} {options}",
            *(["--async-serve"] if async_serve else []),
            "--serve-cmd",
            (
                f"vllm serve {model} {options} --port {get_open_port()} "
                "--kv-cache-memory-bytes 32M"
            ),
            "--num-iters-cold",
            "1",
            "--num-iters-warmup",
            "0",
            "--num-iters-warm",
            "2",
            "--output-json",
            str(output),
        ],
        check=True,
        timeout=900,
    )
    results = json.loads(output.read_text())
    (daemon_ready,) = results["sidecar_ready_times"]
    assert daemon_ready > 0
    warm = results["warm_iterations"]
    assert len(warm) == 2
    if async_serve:
        (pair,) = results["concurrent_iterations"]
        assert 0 <= pair["serve_launch_offset"] < daemon_ready
    else:
        assert "concurrent_iterations" not in results
    for row in [*results.get("concurrent_iterations", []), *warm]:
        assert 0 < row["ready_time"] < row["first_token_time"]
        assert (
            "from the weight cache daemon via zero-copy sharing"
            in Path(row["server_log"]).read_text()
        )


def _run_startup(tmp_path, *options, cold=1, warm=1):
    from vllm.benchmarks.startup import add_cli_args, main
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    output = tmp_path / "result.json"
    args = add_cli_args(FlexibleArgumentParser()).parse_args(
        [
            *options,
            "--num-iters-cold",
            str(cold),
            "--num-iters-warmup",
            "0",
            "--num-iters-warm",
            str(warm),
            "--server-ready-timeout",
            "10",
            "--output-json",
            str(output),
        ]
    )
    main(args)
    return json.loads(output.read_text())


def _file_server(port, directory):
    return shlex.join(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(directory),
        ]
    )


def test_failed_sidecar_setup_stops_warm_starts(tmp_path):
    with pytest.raises(RuntimeError, match="return code 1"):
        _run_startup(
            tmp_path,
            "--pre-serve-cmd",
            "false",
            "--pre-serve-ready-url",
            f"http://127.0.0.1:{get_open_port()}/health",
            "--serve-cmd",
            "false",
            cold=0,
        )
    assert not list(tmp_path.glob("*-logs-*/warm-*.log"))


def test_concurrent_generic_sidecar_can_depend_on_first_request(tmp_path):
    # The sidecar's health URL returns 404 until the server handles a completion.
    # Waiting for sidecar readiness before launching or querying the server deadlocks.
    serve_port, side_port = get_open_ports_list(2)
    signal = tmp_path / "served"
    script = Path(__file__).with_name("startup_completion_server.py")
    results = _run_startup(
        tmp_path,
        "--pre-serve-cmd",
        _file_server(side_port, tmp_path),
        "--async-serve",
        "--pre-serve-ready-url",
        f"http://127.0.0.1:{side_port}/served",
        "--serve-cmd",
        shlex.join(
            [
                sys.executable,
                str(script),
                "--request-log",
                str(signal),
                "--port",
                str(serve_port),
            ]
        ),
        cold=2,
    )
    row, _ = results["concurrent_iterations"]
    assert 0 <= row["serve_launch_offset"] < row["ready_time"] < row["first_token_time"]
    assert row["ready_time"] < results["sidecar_ready_times"][0]
    assert len(results["sidecar_iterations"]) == 2
    first_cold, last_cold, warm = signal.read_text().splitlines()
    assert first_cold != last_cold
    assert warm == last_cold
    for port in (serve_port, side_port):
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()


@pytest.mark.parametrize("failing_process", ["sidecar", "server"])
def test_concurrent_start_cleans_up_on_peer_exit(tmp_path, failing_process):
    # This process stays alive without becoming ready. Its peer exits immediately.
    waiting = shlex.join([sys.executable, "-c", "import signal; signal.pause()"])
    side_port, serve_port = get_open_ports_list(2)
    sidecar = "false" if failing_process == "sidecar" else waiting
    server = "false" if failing_process == "server" else waiting
    with pytest.raises(RuntimeError, match="return code 1"):
        _run_startup(
            tmp_path,
            "--pre-serve-cmd",
            sidecar,
            "--async-serve",
            "--pre-serve-ready-url",
            f"http://127.0.0.1:{side_port}/ready",
            "--serve-cmd",
            f"{server} --port {serve_port}",
        )
    assert not list(tmp_path.glob("*-logs-*/warm-*.log"))


def test_pre_serve_default_only_measures_sidecar_in_cold_phase(tmp_path):
    port = get_open_port()
    result = _run_startup(
        tmp_path,
        "--pre-serve-cmd",
        _file_server(port, tmp_path),
        "--pre-serve-ready-url",
        f"http://127.0.0.1:{port}/",
        "--serve-cmd",
        "false",
        warm=0,
    )
    assert result["sidecar_ready_times"][0] > 0
    assert "concurrent_iterations" not in result
    assert "warm_iterations" not in result
