# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import set_random_seed

if HAS_TRITON:
    from vllm.model_executor.layers.fusion.fused_add_norm import (
        fused_add_layer_norm,
    )

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# N=1152: Qwen3-VL vision hidden; N=1024: Kimi MoonViT hidden.
SHAPES = [
    (1, 128),
    (3, 100),  # non-power-of-2 N, exercises the tail mask
    (256, 1024),
    (4096, 1024),
    (57, 1152),
    (16384, 1152),
]
EPS_VALUES = [1e-6, 1e-5]
HAS_RESIDUAL = [True, False]
HAS_BIAS = [True, False]
SEEDS = [13]

# Outputs are fp32-computed values rounded to the storage dtype, so
# tolerance is a few ULP of the storage dtype.
TOLS = {
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-5, 1e-5),
}


def _ref_fused_add_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_residual = x + residual if residual is not None else x
    normed = F.layer_norm(new_residual, (x.shape[-1],), weight, bias, eps)
    return normed, new_residual


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not HAS_TRITON,
    reason="fused_add_layer_norm Triton kernel requires CUDA/ROCm",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("has_residual", HAS_RESIDUAL)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_fused_add_layer_norm_matches_reference(
    shape: tuple[int, int],
    dtype: torch.dtype,
    has_residual: bool,
    has_bias: bool,
    eps: float,
    seed: int,
):
    set_random_seed(seed)
    device = "cuda"
    M, N = shape

    x = torch.randn(M, N, dtype=dtype, device=device)
    residual = torch.randn(M, N, dtype=dtype, device=device) if has_residual else None
    weight = torch.randn(N, dtype=dtype, device=device) * 0.1 + 1.0
    bias = torch.randn(N, dtype=dtype, device=device) * 0.1 if has_bias else None

    ref_normed, ref_residual = _ref_fused_add_layer_norm(x, residual, weight, bias, eps)
    normed, new_residual = fused_add_layer_norm(x, residual, weight, bias, eps)

    atol, rtol = TOLS[dtype]
    torch.testing.assert_close(normed, ref_normed, atol=atol, rtol=rtol)
    if has_residual:
        # fp32-accumulated add rounded once to the storage dtype, identical
        # to the eager add.
        torch.testing.assert_close(new_residual, ref_residual, atol=0, rtol=0)
    else:
        assert new_residual is x


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not HAS_TRITON,
    reason="fused_add_layer_norm Triton kernel requires CUDA/ROCm",
)
@pytest.mark.parametrize("shape", [(256, 1024), (57, 1152)])
@pytest.mark.parametrize("has_residual", HAS_RESIDUAL)
@torch.inference_mode()
def test_fused_add_layer_norm_fp32_exactness(
    shape: tuple[int, int],
    has_residual: bool,
):
    """fp32 kernel output must sit within fp32 rounding of a fp64 reference."""
    set_random_seed(13)
    device = "cuda"
    M, N = shape

    x = torch.randn(M, N, dtype=torch.float32, device=device)
    residual = (
        torch.randn(M, N, dtype=torch.float32, device=device) if has_residual else None
    )
    weight = torch.randn(N, dtype=torch.float32, device=device) * 0.1 + 1.0
    bias = torch.randn(N, dtype=torch.float32, device=device) * 0.1

    normed, _ = fused_add_layer_norm(x, residual, weight, bias, eps=1e-6)

    base = (x + residual) if residual is not None else x
    ref64 = F.layer_norm(base.double(), (N,), weight.double(), bias.double(), 1e-6)
    torch.testing.assert_close(normed, ref64.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not HAS_TRITON,
    reason="fused_add_layer_norm Triton kernel requires CUDA/ROCm",
)
@pytest.mark.parametrize("has_residual", HAS_RESIDUAL)
def test_fused_add_layer_norm_cuda_graph(has_residual: bool):
    """Kernel launch must be capture-safe and replay identically."""
    set_random_seed(13)
    device = "cuda"
    M, N = 512, 1152
    dtype = torch.bfloat16

    static_x = torch.randn(M, N, dtype=dtype, device=device)
    static_residual = (
        torch.randn(M, N, dtype=dtype, device=device) if has_residual else None
    )
    weight = torch.randn(N, dtype=dtype, device=device) * 0.1 + 1.0
    bias = torch.randn(N, dtype=dtype, device=device) * 0.1

    # Eager call first: compiles the kernel outside graph capture.
    ref_normed, ref_residual = fused_add_layer_norm(
        static_x, static_residual, weight, bias, 1e-6
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_normed, graph_residual = fused_add_layer_norm(
            static_x, static_residual, weight, bias, 1e-6
        )
    graph.replay()

    torch.testing.assert_close(graph_normed, ref_normed, atol=0, rtol=0)
    torch.testing.assert_close(graph_residual, ref_residual, atol=0, rtol=0)
