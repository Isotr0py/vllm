# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from TileKernels (https://github.com/deepseek-ai/TileKernels)
"""Self-contained per-channel FP8 cast kernel with optional token expansion.

Inlines all TileKernels dependencies (utils, quant/common, config) so that
only ``tilelang`` is required as an external package.
"""

from dataclasses import dataclass, replace
from typing import Optional, Union

import tilelang
import tilelang.language as T
import torch
from tilelang.contrib import nvcc
from tilelang.utils.target import determine_target

# ---------------------------------------------------------------------------
# Inline: tile_kernels.utils
# ---------------------------------------------------------------------------


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# ---------------------------------------------------------------------------
# Inline: tile_kernels.quant.types
# ---------------------------------------------------------------------------

QuantTensor = tuple[torch.Tensor, torch.Tensor]

# ---------------------------------------------------------------------------
# Inline: tile_kernels.quant.common  (minimal subset)
# ---------------------------------------------------------------------------


def _get_best_vectorize_size(dtype: T.dtype) -> int:
    target = determine_target(return_object=True)
    ver = nvcc.get_target_compute_version(target)
    major, _ = nvcc.parse_compute_version(ver)
    return (16 if major < 10 else 32) // dtype.bytes


@dataclass(frozen=True)
class _BaseCastConfig:
    torch_dtype: torch.dtype = torch.float8_e4m3fn
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False

    @property
    def dtype(self) -> T.dtype:
        return (
            T.dtype(self.torch_dtype)
            if self.torch_dtype != torch.int8
            else T.float4_e2m1fn
        )

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        return torch.uint8 if self.use_packed_ue8m0 else torch.float32

    @property
    def sf_dtype(self) -> T.dtype:
        return T.dtype(self.sf_torch_dtype)


@dataclass(frozen=True)
class _CastInputConfig(_BaseCastConfig):
    with_sf: bool = True


@dataclass(frozen=True)
class _CastOutputConfig(_BaseCastConfig):
    round_sf: bool = False
    custom_clamp_min_value: float | None = None

    @property
    def clamp_min_value(self) -> float:
        if self.custom_clamp_min_value is not None:
            return self.custom_clamp_min_value
        elif self.dtype == T.float8_e4m3fn:
            return 1e-4
        elif self.dtype == T.float4_e2m1fn:
            return T.max_value(self.dtype) * (2**-126)
        else:
            raise ValueError(f"Unsupported dtype {self.dtype}")


def _get_cast_input_and_config(
    x: Union[torch.Tensor, QuantTensor],
    sf_block: Optional[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor | None, _CastInputConfig]:
    if isinstance(x, tuple):
        assert isinstance(sf_block, tuple)
        x, x_sf = x
        config = _CastInputConfig(
            torch_dtype=x.dtype, with_sf=True, sf_block=sf_block
        )
        assert isinstance(x, torch.Tensor) and isinstance(x_sf, torch.Tensor)
        assert x.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)

        if x_sf.stride(0) == 1:
            config = replace(config, use_tma_aligned_col_major_sf=True)
            x_sf = x_sf.T
            if x_sf.dtype == torch.int32:
                config = replace(config, use_packed_ue8m0=True)
                x_sf = x_sf.view(torch.uint8)
        else:
            assert x_sf.stride(1) == 1
            assert x_sf.dtype == torch.float32
        return x, x_sf, config
    else:
        config = _CastInputConfig(torch_dtype=x.dtype, with_sf=False)
        assert sf_block is None
        assert isinstance(x, torch.Tensor)
        assert x.dtype in (torch.bfloat16, torch.float32)
        return x, None, config


def _get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    custom_clamp_min_value: float | None = None,
) -> _CastOutputConfig:
    assert fmt in ("e5m6", "e4m3", "e2m1")
    mapping = {
        "e5m6": torch.uint32,
        "e4m3": torch.float8_e4m3fn,
        "e2m1": torch.int8,
    }
    return _CastOutputConfig(
        torch_dtype=mapping[fmt],
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        custom_clamp_min_value=custom_clamp_min_value,
    )


# ---------------------------------------------------------------------------
# TileLang macros
# ---------------------------------------------------------------------------


@T.macro
def _get_sf_and_inv(amax: float, out_config: _CastOutputConfig):
    clamped_amax = T.max(amax, out_config.clamp_min_value)
    max_value = T.max_value(out_config.dtype)
    sf = clamped_amax / max_value
    if not out_config.round_sf:
        return sf, max_value / clamped_amax
    bits = T.reinterpret(sf, T.uint32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
    if out_config.use_packed_ue8m0:
        return T.uint8(exp_sf + 127), sf_inv
    else:
        return T.reinterpret((127 + exp_sf) << 23, T.float32), sf_inv


# ---------------------------------------------------------------------------
# JIT kernel: per_channel_cast_fused
# ---------------------------------------------------------------------------


def _transform_token_idx(with_expand: bool, idx: int, token_idx: int, x):
    if with_expand:
        return x[idx]
    return token_idx


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _get_per_channel_cast_fused_kernel(
    hidden: int,
    with_expand: bool,
    in_config: _CastInputConfig,
    out_config: _CastOutputConfig,
):
    num_tokens = T.dynamic("num_tokens")
    num_tokens_out = T.dynamic("num_tokens_out")

    num_per_tokens, _ = out_config.sf_block
    _, num_per_channels = in_config.sf_block
    assert num_per_tokens == 128
    assert not in_config.with_sf or num_per_channels == 128

    num_threads = 256
    TILE_M, TILE_K = 128, 128
    if in_config.with_sf:
        TILE_K = 256

    num_threads_per_token = 32
    assert TILE_K % num_threads_per_token == 0

    VEC_K = TILE_K // num_threads_per_token
    VEC_M = TILE_M * num_threads_per_token // num_threads

    @T.prim_func
    def per_channel_cast_fused_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens_out, hidden), out_config.dtype],
        out_sf: T.Tensor[
            (T.ceildiv(num_tokens_out, num_per_tokens), hidden), out_config.sf_dtype
        ],
        x_sf_invs: T.Tensor[
            (num_tokens, T.ceildiv(hidden, num_per_channels)), in_config.sf_dtype
        ],
        pos_to_token: T.Tensor[(num_tokens_out,), T.int32],
    ):
        with T.Kernel(
            T.ceildiv(num_tokens_out, TILE_M),
            T.ceildiv(hidden, TILE_K),
            threads=num_threads,
        ) as (pid_token, pid_hidden):
            x_shared = T.alloc_shared((TILE_M, TILE_K), in_config.dtype)
            pos_to_token_local = T.alloc_local((VEC_M,), T.int32)
            sf_invs_local = T.alloc_local((VEC_M,), T.float32)
            amax_local = T.alloc_local((VEC_K,), T.float32)
            amax_shared = T.alloc_shared((VEC_K, num_threads), T.float32)
            in_local = T.alloc_local((VEC_K,), in_config.dtype)
            out_local = T.alloc_local((VEC_K,), out_config.dtype)
            tid = T.get_thread_binding(0)
            m_id, k_id = tid // num_threads_per_token, tid % num_threads_per_token
            m_offset = pid_token * TILE_M + m_id * VEC_M
            k_offset = pid_hidden * TILE_K + k_id * VEC_K

            T.assume(
                num_tokens_out % 128 == 0
                or (with_expand and num_tokens_out % 16 == 0)
            )
            if with_expand:
                tmp = T.alloc_var(T.int32)
                if k_id < VEC_M:
                    tmp = pos_to_token[k_id + m_offset]

                for i in T.serial(VEC_M):
                    pos_to_token_local[i] = T.shfl_sync(tmp, i)

            if in_config.with_sf:
                for i in T.serial(VEC_M):
                    pos = _transform_token_idx(
                        with_expand, i, i + m_offset, pos_to_token_local
                    )
                    T.assume(pos < num_tokens)
                    sf_invs_local[i] = T.Select(
                        with_expand and pos < 0,
                        0.0,
                        x_sf_invs[
                            pos,
                            (pid_hidden * TILE_K + k_id * VEC_K) // num_per_channels,
                        ],
                    )

            T.clear(amax_local)
            for i in T.serial(VEC_M):
                pos = _transform_token_idx(
                    with_expand, i, i + m_offset, pos_to_token_local
                )
                T.assume(pos < num_tokens)
                if not with_expand or pos >= 0:
                    for j in T.vectorized(VEC_K):
                        T.assume(pos < num_tokens)
                        in_local[j] = x[pos, j + k_offset]
                        x_shared[i + m_id * VEC_M, j + k_id * VEC_K] = in_local[j]

                    for j in T.vectorized(VEC_K):
                        if in_config.with_sf:
                            amax_local[j] = T.max(
                                amax_local[j], T.abs(in_local[j] * sf_invs_local[i])
                            )
                        else:
                            amax_local[j] = T.max(
                                amax_local[j], T.abs(in_local[j])
                            )
                else:
                    for j in T.vectorized(VEC_K):
                        x_shared[i + m_id * VEC_M, j + k_id * VEC_K] = 0

            for i in T.unroll(VEC_K):
                amax_shared[i, tid] = amax_local[i]

            sf = T.alloc_var(T.float32)
            sf = 0
            col_id = tid % num_threads_per_token * VEC_K + tid // num_threads_per_token
            if tid < TILE_K:
                for i in T.serial(col_id // VEC_K, num_threads, num_threads_per_token):
                    sf = T.max(sf, amax_shared[col_id % VEC_K, i])

                sf, sf_inv = _get_sf_and_inv(sf, out_config)
                out_sf[pid_token, pid_hidden * TILE_K + col_id] = sf
                amax_shared[0, tid] = sf_inv

            for i in T.serial(VEC_K):
                amax_local[i] = amax_shared[0, k_id + i * num_threads_per_token]

            for i in T.serial(VEC_M):
                for j in T.vectorized(VEC_K):
                    in_local[j] = x_shared[i + m_id * VEC_M, j + k_id * VEC_K]
                for j in T.vectorized(VEC_K):
                    if in_config.with_sf:
                        out_local[j] = in_local[j] * sf_invs_local[i] * amax_local[j]
                    else:
                        out_local[j] = in_local[j] * amax_local[j]
                for j in T.vectorized(VEC_K):
                    out[i + m_offset, j + k_offset] = out_local[j]

    return per_channel_cast_fused_kernel


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def per_channel_cast_fused(
    x: torch.Tensor,
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
    x_sf_invs: torch.Tensor | None = None,
    num_per_channels: int | None = None,
    pos_to_token: torch.Tensor | None = None,
) -> QuantTensor:
    """Cast a matrix to FP8 with per-channel scaling, optionally fusing
    rescaling and token expansion.

    Args:
        x: Input tensor of shape ``(num_tokens, hidden)`` in bf16/fp32,
            or FP8 (float8_e4m3fn) when ``x_sf_invs`` is provided.
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_tokens: Number of tokens in each scaling block (must be 128).
        round_sf: Whether to round scaling factors to powers of two.
        x_sf_invs: Optional inverse scaling factors for FP8 input rescaling.
            Shape ``(num_tokens, ceil(hidden, num_per_channels))``.
        num_per_channels: Number of channels in each input scaling block.
            Required when ``x_sf_invs`` is provided (must be 128).
        pos_to_token: Optional int32 index tensor for token expansion/gather.
            Shape ``(num_tokens_out,)``.  Negative values indicate masked-out
            positions (output padded with zeros).

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and per-channel sf tensor.
    """
    input_tuple: QuantTensor | torch.Tensor
    if x_sf_invs is not None:
        input_tuple = (x, x_sf_invs)
    else:
        input_tuple = x

    x, x_sf_invs, in_config = _get_cast_input_and_config(
        input_tuple,
        None if num_per_channels is None else (1, num_per_channels),
    )

    assert fmt == "e4m3"
    assert x.dim() == 2 and x.is_contiguous()
    num_tokens, hidden = x.shape
    num_tokens_out = num_tokens

    if pos_to_token is not None:
        assert pos_to_token.dim() == 1 and pos_to_token.is_contiguous()
        assert pos_to_token.dtype == torch.int32
        num_tokens_out = pos_to_token.size(0)
        assert num_tokens_out % 16 == 0
    else:
        assert num_tokens_out % 128 == 0

    assert num_per_tokens == 128
    if x_sf_invs is not None:
        assert num_per_channels == 128
        assert x.dtype == torch.float8_e4m3fn
        assert x_sf_invs.dim() == 2 and x_sf_invs.is_contiguous()
        assert (
            x_sf_invs.size(0) == num_tokens and x_sf_invs.size(1) * 128 == hidden
        )

    out_config = _get_cast_output_config(fmt, (num_per_tokens, 1), round_sf=round_sf)
    kernel = _get_per_channel_cast_fused_kernel(
        hidden,
        with_expand=(pos_to_token is not None),
        in_config=in_config,
        out_config=out_config,
    )

    out = torch.empty(
        (num_tokens_out, hidden), dtype=out_config.torch_dtype, device="cuda"
    )
    out_sf = torch.empty(
        (_ceil_div(num_tokens_out, num_per_tokens), hidden),
        dtype=torch.float32,
        device="cuda",
    )
    if num_tokens_out > 0:
        kernel(x, out, out_sf, x_sf_invs, pos_to_token)

    return out, out_sf


# ---------------------------------------------------------------------------
# Custom op registration (torch.ops.vllm.per_channel_cast_fused)
# ---------------------------------------------------------------------------


def _per_channel_cast_fused_fake(
    x: torch.Tensor,
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
    x_sf_invs: torch.Tensor | None = None,
    num_per_channels: int | None = None,
    pos_to_token: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens_out = (
        pos_to_token.size(0) if pos_to_token is not None else x.size(0)
    )
    hidden = x.size(1)
    out = torch.empty(
        num_tokens_out, hidden, dtype=torch.float8_e4m3fn, device=x.device
    )
    out_sf = torch.empty(
        _ceil_div(num_tokens_out, num_per_tokens),
        hidden,
        dtype=torch.float32,
        device=x.device,
    )
    return out, out_sf


from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    op_name="per_channel_cast_fused",
    op_func=per_channel_cast_fused,
    mutates_args=[],
    fake_impl=_per_channel_cast_fused_fake,
)
