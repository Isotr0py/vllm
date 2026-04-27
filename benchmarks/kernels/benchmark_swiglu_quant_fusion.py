# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: separate SiluAndMul + FP8 quant vs fused swiglu+quant.

Justifies whether extending the linear layer interface for
pre-quantized input is worthwhile.

Usage:
    python benchmarks/kernels/benchmark_swiglu_quant_fusion.py
"""

import argparse
import time

import torch
from vllm._custom_ops import scaled_fp8_quant

from vllm.model_executor.layers.fused_moe.ops.swiglu_fp8_quant_kernel import (
    swiglu_forward_and_per_token_cast,
)

# vLLM's CUDA SiluAndMul kernel (torch.ops._C.silu_and_mul)
_silu_and_mul_cuda = torch.ops._C.silu_and_mul


def bench(fn, warmup=10, iters=100) -> float:
    """Return median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=7168,
                        help="Intermediate size (after SwiGLU split)")
    parser.add_argument("--clamp", type=float, default=None,
                        help="SwiGLU clamp value (None = no clamp)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    hidden = args.hidden  # output hidden after SwiGLU (input is 2*hidden)
    clamp_val = args.clamp

    # Per-token quantization via scaled_fp8_quant (matches V4 FP8 path)
    token_counts = [128, 256, 512, 1024, 2048, 4096]

    label = (f"SwiGLU+Quant Fusion Benchmark  "
             f"hidden={hidden}, clamp={clamp_val}")
    print(label)
    print()
    print(f"{'Tokens':>8s} | {'Baseline (2 kernels)':>22s} | "
          f"{'Fused (1 kernel)':>22s} | {'Savings':>8s}")
    print("-" * 76)

    for num_tokens in token_counts:
        gate_up = torch.randn(num_tokens, hidden * 2, dtype=torch.bfloat16,
                              device="cuda")

        # ---- Baseline: vLLM CUDA SiluAndMul + scaled_fp8_quant (2 kernels) ----
        swiglu_buf = torch.empty(num_tokens, hidden, dtype=torch.bfloat16,
                                 device="cuda")

        def run_baseline():
            _silu_and_mul_cuda(swiglu_buf, gate_up)          # CUDA SiluAndMul
            fp8_out, scale = scaled_fp8_quant(swiglu_buf)     # FP8 quant

        t_base = bench(run_baseline, args.warmup, args.iters)

        # ---- Fused: swiglu_forward_and_per_token_cast (one kernel) ----
        def run_fused():
            fp8_out, scale = swiglu_forward_and_per_token_cast(
                x=gate_up,
                fmt="e4m3",
                num_per_channels=128,
                swiglu_clamp_value=clamp_val,
            )

        t_fused = bench(run_fused, args.warmup, args.iters)

        savings = (t_base - t_fused) / t_base * 100
        print(f"{num_tokens:>8d} | {t_base:>19.1f} us | {t_fused:>19.1f} us | "
              f"{savings:>+6.1f}%")


if __name__ == "__main__":
    main()
