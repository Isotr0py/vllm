# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure engine or server startup with cold and warm compilation caches."""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from vllm.benchmarks.lib.startup import cold_startup, measure_server_startup
from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_env_var

PERCENTAGES = [10, 25, 50, 75, 90, 99]


def _start_engine(args: EngineArgs) -> dict[str, float]:
    from vllm import LLM

    start = time.perf_counter()
    llm = LLM.from_engine_args(args)
    elapsed = time.perf_counter() - start
    config = llm.llm_engine.vllm_config.compilation_config
    return {
        "startup_time": elapsed,
        "compilation_time": config.compilation_time,
        "encoder_compilation_time": config.encoder_compilation_time,
    }


def _measure_engine_startup(
    args: argparse.Namespace,
) -> dict[str, list[dict[str, float]]]:
    engine_args = EngineArgs.from_cli_args(args)
    spawn = multiprocessing.get_context("spawn")
    iterations: dict[str, list[dict[str, float]]] = {}
    with set_env_var("VLLM_ENABLE_V1_MULTIPROCESSING", "0"):
        for phase, count in (
            ("cold", args.num_iters_cold),
            ("warmup", args.num_iters_warmup),
            ("warm", args.num_iters_warm),
        ):
            rows = iterations[phase] = []
            for _ in tqdm(range(count), desc=f"{phase.capitalize()} engine starts"):
                with (
                    cold_startup() if phase == "cold" else nullcontext(),
                    ProcessPoolExecutor(max_workers=1, mp_context=spawn) as worker,
                ):
                    rows.append(worker.submit(_start_engine, engine_args).result())
    return iterations


def _report(args: argparse.Namespace, iterations: dict[str, list[dict[str, Any]]]):
    results: dict[str, Any] = {}
    output = Path(args.output_json) if args.output_json else None
    # PyTorch records identify the model advertised by the running server.
    model = next(
        (
            row["model"]
            for rows in iterations.values()
            for row in rows
            if "model" in row
        ),
        None if args.serve_cmd else args.model,
    )
    export_args = argparse.Namespace(**(vars(args) | {"model": model}))
    has_encoder = any(
        row.get("encoder_compilation_time", 0)
        for phase in ("cold", "warm")
        for row in iterations.get(phase, [])
    )

    for phase, rows in iterations.items():
        if args.serve_cmd:
            results[f"{phase}_iterations"] = rows
        if not rows or phase in ("warmup", "sidecar_setup"):
            continue
        print(f"\n{phase.upper()} STARTUP ({len(rows)} runs)")
        for field in rows[0]:
            if not field.endswith("_time"):
                continue
            metric = field.removesuffix("_time")
            if metric == "encoder_compilation" and not has_encoder:
                continue
            values = [row[field] for row in rows]
            samples = np.array(values)
            mean = float(samples.mean())
            percentiles = {p: float(np.percentile(samples, p)) for p in PERCENTAGES}
            key = f"{phase}_{metric}"
            stats = {f"{key}_times": values, f"{key}_percentiles": percentiles}
            results.update(stats)
            results[f"avg_{key}_time"] = mean
            distribution = ", ".join(f"p{p}={v:.2f}s" for p, v in percentiles.items())
            print(f"  {metric}: avg={mean:.2f}s, {distribution}")
            if output and model is not None:
                records = convert_to_pytorch_benchmark_format(
                    export_args, {f"avg_{key}_time": [mean]}, stats
                )
                if records:
                    write_to_json(
                        f"{output.with_suffix('')}.{key}.pytorch.json", records
                    )
    if output:
        output.write_text(json.dumps(results, indent=4))


def add_cli_args(parser: FlexibleArgumentParser):
    parser.add_argument(
        "--num-iters-cold",
        type=int,
        default=3,
        help="Number of cold startup iterations.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=1,
        help="Number of warmup iterations before benchmarking warm startups.",
    )
    parser.add_argument(
        "--num-iters-warm",
        type=int,
        default=3,
        help="Number of warm startup iterations.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the startup time results in JSON format.",
    )

    parser.add_argument(
        "--serve-cmd",
        help="Time a server command from launch to readiness and to the first "
        "non-empty text in a streamed completion. Put all engine arguments inside "
        "the quoted command.",
    )
    parser.add_argument(
        "--pre-serve-cmd",
        help="Long-running command to start in each cold iteration. By default, "
        "time its readiness, then reuse the last process for warm server starts.",
    )
    parser.add_argument(
        "--pre-serve-ready-url",
        help="HTTP endpoint that returns 200 when the sidecar is ready. With this "
        "option both commands run as supplied. For vllm preload, omit it to "
        "configure a private IPC socket and health endpoint automatically.",
    )
    parser.add_argument(
        "--async-serve",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also launch the server in each cold iteration, immediately after "
        "--pre-serve-cmd. Probe both processes independently.",
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for each server or sidecar to become ready. "
        "A server that exits is reported at once.",
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser


def main(args: argparse.Namespace):
    if min(args.num_iters_cold, args.num_iters_warmup, args.num_iters_warm) < 0:
        raise ValueError("Iteration counts must be nonnegative")
    if args.pre_serve_cmd and not args.serve_cmd:
        raise ValueError("--pre-serve-cmd requires --serve-cmd")
    if (args.pre_serve_ready_url or args.async_serve) and not args.pre_serve_cmd:
        raise ValueError(
            "--pre-serve-ready-url and --async-serve require --pre-serve-cmd"
        )
    if args.serve_cmd:
        # Sweep also uses this CLI. An engine override outside --serve-cmd would
        # label a sweep point with a configuration that the server never ran.
        defaults = EngineArgs.add_cli_args(FlexibleArgumentParser()).parse_args([])
        for name, value in vars(defaults).items():
            if getattr(args, name) != value:
                raise ValueError(f"Put --{name.replace('_', '-')} inside --serve-cmd")
        iterations = measure_server_startup(args)
    else:
        iterations = _measure_engine_startup(args)
    _report(args, iterations)
