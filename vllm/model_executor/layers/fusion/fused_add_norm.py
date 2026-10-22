# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused residual-add + LayerNorm Triton kernel.

Equivalent to::

    new_residual = x + residual  (or x when residual is None)
    normed = layer_norm(new_residual, weight, bias, eps)

The residual sum is rounded to the storage dtype before normalization and
normalization runs in fp32, matching eager ``nn.LayerNorm(x + residual)``
numerics.
"""

import torch

from vllm.triton_utils import tl, triton


# num_rows is not specialized: token/patch counts vary per request, and a
# recompile must never trigger inside encoder CUDA graph capture. All
# remaining specialization keys (N, BLOCK_N, the HAS_* flags, num_warps,
# dtype) are fixed per model, so the first eager call compiles every
# variant the graph will ever launch.
@triton.jit(do_not_specialize=["num_rows"])
def _fused_add_layer_norm_kernel(
    x_ptr,
    res_ptr,
    normed_ptr,
    res_out_ptr,
    w_ptr,
    b_ptr,
    num_rows,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        s = x.to(tl.float32) + tl.load(res_ptr + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        # Round the sum back to the storage dtype before normalizing, matching
        # eager nn.LayerNorm(x + residual) numerics; a no-op for fp32 inputs.
        x = s.to(x_ptr.dtype.element_ty)
        tl.store(res_out_ptr + offs, x, mask=mask)

    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    xc = tl.where(mask, xf - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / N + eps)

    y = xc * rstd * tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        y += tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(normed_ptr + offs, y.to(normed_ptr.dtype.element_ty), mask=mask)


def fused_add_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + LayerNorm.

    Args:
        x: Input tensor, shape ``(M, N)``, contiguous.
        residual: Residual tensor with the same shape/dtype as ``x``, or
            ``None`` for a plain LayerNorm.
        weight: LayerNorm scale, shape ``(N,)``.
        bias: LayerNorm shift, shape ``(N,)``, or ``None`` (bias-free norm).
        eps: LayerNorm epsilon.

    Returns:
        ``(normed, new_residual)`` where ``new_residual = x + residual`` in
        ``x.dtype`` and ``normed = layer_norm(new_residual)``. With
        ``residual=None``, returns ``(layer_norm(x), x)`` without copying.

    CUDA-graph safe: no host sync or data-dependent branching. Inputs must
    be 2D contiguous with 16-byte-aligned base pointers (any fresh
    allocation from the torch caching allocator qualifies); Triton's
    pointer-alignment specialization keys on them.

    """
    assert x.ndim == 2 and x.is_contiguous(), (
        f"x must be 2D contiguous, got shape {tuple(x.shape)}"
    )
    num_rows, N = x.shape
    assert weight.numel() == N, f"weight must have {N} elements"
    assert bias is None or bias.numel() == N, f"bias must have {N} elements"
    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == x.dtype, (
            f"residual shape/dtype {tuple(residual.shape)}/{residual.dtype} "
            f"!= x {tuple(x.shape)}/{x.dtype}"
        )
        assert residual.is_contiguous(), "residual must be contiguous"
    # int32 row offsets (row * N) for address arithmetic; guard the range.
    assert num_rows * N < 2**31, f"numel={num_rows * N} exceeds the int32 index range"

    normed = torch.empty_like(x)
    new_residual = torch.empty_like(x) if residual is not None else x

    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N >= 2048:
        num_warps = 8
    elif BLOCK_N >= 512:
        num_warps = 4
    else:
        num_warps = 2

    _fused_add_layer_norm_kernel[(num_rows,)](
        x,
        residual if residual is not None else x,
        normed,
        new_residual,
        weight,
        bias if bias is not None else weight,
        num_rows,
        eps,
        N=N,
        BLOCK_N=BLOCK_N,
        HAS_RESIDUAL=residual is not None,
        HAS_BIAS=bias is not None,
        num_warps=num_warps,
    )
    return normed, new_residual
