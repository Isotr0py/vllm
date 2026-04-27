# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: fused_topk_bias (vLLM CUDA) vs top2_sum_gate (TileLang).

Usage:
    python benchmarks/kernels/benchmark_top2_sum_gate.py

Typical DeepSeek-V4 routing params:
    n_routed_experts=256, num_experts_per_tok=8, scoring_func="sqrtsoftplus"
"""

import argparse
import time

import torch

# Trigger custom-op registration
import vllm.model_executor.layers.fused_moe.ops.top2_sum_gate_kernel  # noqa: F401
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)


def bench(
    label: str,
    fn: callable,
    warmup: int = 10,
    iters: int = 100,
) -> float:
    """Return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us

    times.sort()
    median = times[len(times) // 2]
    return median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--scaling-factor", type=float, default=1.0)
    parser.add_argument(
        "--with-bias", action="store_true", help="Use non-zero e_score_correction_bias"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = "cuda"
    num_experts = args.num_experts
    top_k = args.top_k
    scaling_factor = args.scaling_factor
    renormalize = True

    token_counts = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]

    print("DeepSeek-V4 MoE Routing Benchmark")
    print(
        f"  num_experts={num_experts}, top_k={top_k}, "
        f"scoring=sqrtsoftplus, scaling={scaling_factor}"
    )
    print(f"  with_bias={args.with_bias}, renormalize={renormalize}")
    print()
    print(
        f"{'Tokens':>8s} | {'fused_topk_bias':>18s} | {'top2_sum_gate':>18s} | "
        f"{'Speedup':>8s}"
    )
    print("-" * 68)

    for num_tokens in token_counts:
        # Pad to multiple of 128 for top2_sum_gate alignment
        num_tokens_padded = ((num_tokens + 127) // 128) * 128

        logits = torch.randn(
            num_tokens_padded, num_experts, dtype=torch.float32, device=device
        )
        bias_tensor = (
            torch.randn(num_experts, dtype=torch.float32, device=device) * 0.1
            if args.with_bias
            else torch.zeros(num_experts, dtype=torch.float32, device=device)
        )

        # ---- fused_topk_bias (existing) ----
        def run_fused():
            fused_topk_bias(
                hidden_states=torch.empty(
                    num_tokens_padded, 1, dtype=torch.float16, device=device
                ),
                gating_output=logits,
                scoring_func="sqrtsoftplus",
                e_score_correction_bias=bias_tensor if args.with_bias else None,
                topk=top_k,
                renormalize=renormalize,
                routed_scaling_factor=scaling_factor,
            )

        t_fused = bench("fused_topk_bias", run_fused, args.warmup, args.iters)

        # ---- top2_sum_gate (TileLang) ----
        def run_tk():
            torch.ops.vllm.top2_sum_gate(
                logits=logits,
                bias=bias_tensor,
                num_topk=top_k,
                num_topk_groups=0,
                num_groups=0,
                use_shared_as_routed=False,
                num_shared_experts=0,
                routed_scaling_factor=scaling_factor,
                ep_rank=0,
                num_ep_ranks=1,
                tp_rank=0,
                num_tp_ranks=1,
                scoring_func="sqrtsoftplus",
            )

        t_tk = bench("top2_sum_gate", run_tk, args.warmup, args.iters)

        speedup = t_fused / t_tk if t_tk > 0 else float("inf")
        print(
            f"{num_tokens:>8d} | {t_fused:>15.1f} us | {t_tk:>15.1f} us | "
            f"{speedup:>7.2f}x"
        )


if __name__ == "__main__":
    main()
