# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: TileLang vs Triton SwiGLU + FP8 quantization kernels.

Compares kernels used in DeepSeek V4 MoE _act_mul_quant:
  1. TileLang: torch.ops.vllm.swiglu_fp8_quant
  2. Triton col-major (float32):    silu_mul_per_token_group_quant_fp8_colmajor
  3. Triton col-major (ceil UE8M0): silu_mul_per_token_group_quant_fp8_colmajor(use_ue8m0=True)
  4. Triton UE8M0 packed:           silu_mul_quant_fp8_packed_triton

Note: FP8 (e4m3fn) requires SM90+ (Hopper). On SM86 (Ampere), only TileLang
kernels will work since they generate FP8 PTX directly. Triton kernels will
be skipped gracefully.

Usage:
    python benchmarks/kernels/benchmark_tilelang_vs_triton_swiglu_fp8.py
"""

import numpy as np
import torch

import vllm.model_executor.layers.fused_moe.ops.swiglu_fp8_quant_kernel  # noqa: F401
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_per_token_group_quant_fp8_colmajor,
    silu_mul_quant_fp8_packed_triton,
)
from vllm.utils.import_utils import has_tilelang

# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

GROUP_SIZE = 128
WARMUP_ITERS = 20
BENCH_ITERS = 100
REPEATS_PER_ITER = 5


def _benchmark_kernel(fn, *args, warmup=WARMUP_ITERS, iters=BENCH_ITERS,
                      repeats=REPEATS_PER_ITER):
    """Benchmark a kernel call, return median latency in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    latencies = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeats):
            fn(*args)
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) / repeats)

    return float(np.median(latencies))


# ---------------------------------------------------------------------------
# Kernel wrappers (uniform interface: x -> (out, out_sf))
# ---------------------------------------------------------------------------

def _tilelang_colmajor(x):
    return torch.ops.vllm.swiglu_fp8_quant(
        x=x, fmt="e4m3", num_per_channels=GROUP_SIZE,
        use_tma_aligned_col_major_sf=True,
    )


def _tilelang_ue8m0(x):
    return torch.ops.vllm.swiglu_fp8_quant(
        x=x, fmt="e4m3", num_per_channels=GROUP_SIZE,
        use_packed_ue8m0=True,
    )


def _triton_colmajor(x):
    return silu_mul_per_token_group_quant_fp8_colmajor(
        input=x, use_ue8m0=False,
    )


def _triton_colmajor_ceil(x):
    return silu_mul_per_token_group_quant_fp8_colmajor(
        input=x, use_ue8m0=True,
    )


def _triton_packed(x):
    return silu_mul_quant_fp8_packed_triton(
        input=x, group_size=GROUP_SIZE,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"tilelang available: {has_tilelang()}")
    print()

    KERNELS = [
        ("TileLang-colmajor", _tilelang_colmajor, has_tilelang()),
        ("TileLang-UE8M0",    _tilelang_ue8m0,    has_tilelang()),
        ("Triton-colmajor",   _triton_colmajor,    True),
        ("Triton-ceil",       _triton_colmajor_ceil, True),
        ("Triton-packed",     _triton_packed,      True),
    ]

    # Probe which kernels actually work
    available_kernels = []
    probe_x = torch.randn(128, 7168 * 2, dtype=torch.bfloat16, device="cuda")
    for name, fn, cond in KERNELS:
        if not cond:
            continue
        try:
            fn(probe_x)
            available_kernels.append((name, fn))
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [SKIP] {name}: {type(e).__name__}: {e}")
    print()

    if len(available_kernels) < 2:
        print("Need at least 2 available kernels for comparison. "
              "FP8 Triton kernels require SM90+ (Hopper).")
        if len(available_kernels) == 1:
            print("\nShowing single available kernel performance:")
        else:
            return

    # M values to sweep (num tokens / expanded tokens after top-k)
    M_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    H_values = [7168]  # DS-V3/V4 intermediate

    for H in H_values:
        print(f"\n{'='*100}")
        print(f"  Hidden = {H}  |  Group size = {GROUP_SIZE}")
        print(f"  Input shape: (M, {H*2}) bf16  ->  Output: (M, {H}) fp8 + scales")
        print(f"{'='*100}")

        # Print header
        header = f"{'M':>6}"
        for name, _ in available_kernels:
            header += f" | {name:>18s}"
        if len(available_kernels) >= 2:
            header += f" | {'Speedup':>16s}"
        print(header)
        print("-" * len(header))

        for M in M_values:
            M_aligned = ((M + 7) // 8) * 8
            x = torch.randn(M_aligned, H * 2, dtype=torch.bfloat16, device="cuda")

            results = {}
            for name, fn in available_kernels:
                try:
                    latency_ms = _benchmark_kernel(fn, x)
                    input_bytes = M_aligned * H * 2 * 2
                    output_bytes = M_aligned * H * 1
                    scale_bytes = M_aligned * (H // GROUP_SIZE) * 4
                    total_bytes = input_bytes + output_bytes + scale_bytes
                    gbps = total_bytes / (latency_ms * 1e-3) / 1e9
                    results[name] = (latency_ms, gbps)
                except Exception as e:
                    results[name] = (float('nan'), 0.0)

            # Format row
            row = f"{M:>6d}"
            for name, _ in available_kernels:
                if name in results:
                    ms, gbps = results[name]
                    row += f" | {ms*1e3:>10.1f}us {gbps:>5.0f}G"
                else:
                    row += f" | {'N/A':>18s}"

            # Speedup column: first TileLang vs first Triton
            if len(available_kernels) >= 2:
                tl_names = [n for n, _ in available_kernels if "TileLang" in n]
                tr_names = [n for n, _ in available_kernels if "Triton" in n]
                if tl_names and tr_names:
                    tl_ms = results.get(tl_names[0], (float('nan'),))[0]
                    tr_ms = results.get(tr_names[0], (float('nan'),))[0]
                    speedup = tr_ms / tl_ms if tl_ms > 0 and not np.isnan(tl_ms) else float('nan')
                    row += f" | {speedup:>12.2f}x"
                else:
                    row += f" | {'':>16s}"

            print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
