# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from TileKernels (https://github.com/deepseek-ai/TileKernels)
"""Self-contained SwiGLU + per-token FP8 quantization kernel.

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


def _align(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ---------------------------------------------------------------------------
# Inline: tile_kernels.config
# ---------------------------------------------------------------------------

import functools


@functools.lru_cache(maxsize=None)
def _get_device_num_sms() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.multi_processor_count


def _get_num_sms() -> int:
    return _get_device_num_sms()


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
        return T.dtype(self.torch_dtype) if self.torch_dtype != torch.int8 else T.float4_e2m1fn

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        return torch.uint8 if self.use_packed_ue8m0 else torch.float32

    @property
    def sf_dtype(self) -> T.dtype:
        return T.dtype(self.sf_torch_dtype)


@dataclass(frozen=True)
class _CastOutputConfig(_BaseCastConfig):
    round_sf: bool = False
    custom_clamp_min_value: Optional[float] = None

    @property
    def clamp_min_value(self) -> float:
        if self.custom_clamp_min_value is not None:
            return self.custom_clamp_min_value
        elif self.dtype == T.float8_e4m3fn:
            return 1e-4
        elif self.dtype == T.float4_e2m1fn:
            return T.max_value(self.dtype) * (2**-126)
        else:
            raise ValueError(f'Unsupported dtype {self.dtype}')


def _get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    custom_clamp_min_value: Optional[float] = None,
) -> _CastOutputConfig:
    assert fmt in ('e5m6', 'e4m3', 'e2m1')
    mapping = {
        'e5m6': torch.uint32,
        'e4m3': torch.float8_e4m3fn,
        'e2m1': torch.int8,
    }
    return _CastOutputConfig(
        torch_dtype=mapping[fmt],
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        custom_clamp_min_value=custom_clamp_min_value,
    )


def _get_sf_shape(shape: tuple[int, int], config: _BaseCastConfig) -> tuple[int, int]:
    num_block_m = _ceil_div(shape[0], config.sf_block[0])
    num_block_k = _ceil_div(shape[1], config.sf_block[1])
    if config.use_packed_ue8m0:
        num_block_m = num_block_m * 4
        num_block_k = _ceil_div(num_block_k, 4)
    return (num_block_k, num_block_m) if config.use_tma_aligned_col_major_sf else (num_block_m, num_block_k)


def _alloc_scaling_factors(
    shape: tuple[int, int],
    out_config: _BaseCastConfig,
    device: torch.device = 'cuda',
) -> torch.Tensor:
    sf_shape = _get_sf_shape(shape, out_config)
    aligned_sf_shape = sf_shape[1]
    if out_config.use_tma_aligned_col_major_sf:
        aligned_sf_shape = _align(sf_shape[1], 16 if out_config.use_packed_ue8m0 else 4)
    scaling_factor = torch.empty(
        size=(sf_shape[0], aligned_sf_shape),
        dtype=out_config.sf_torch_dtype,
        device=device,
    )
    if out_config.use_tma_aligned_col_major_sf:
        scaling_factor = scaling_factor[:, : sf_shape[1]]
    return scaling_factor


def _cast_epilogue(
    out_sf: torch.Tensor,
    num_tokens: int,
    hidden: int,
    config: _BaseCastConfig,
) -> torch.Tensor:
    if config.use_packed_ue8m0:
        if num_tokens == 0:
            out_sf = torch.empty(
                (out_sf.shape[0], out_sf.shape[1] // 4),
                dtype=torch.int32,
                device=out_sf.device,
            )
        else:
            out_sf = out_sf.view(dtype=torch.int32)
    out_sf = out_sf.T if config.use_tma_aligned_col_major_sf else out_sf
    out_sf = out_sf[: _ceil_div(num_tokens, config.sf_block[0]), :]
    return out_sf


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


@T.macro
def _store_sf(tensor: T.Tensor, sf, m_idx: int, k_idx: int, config: _BaseCastConfig):
    if config.use_packed_ue8m0:
        tensor[k_idx // 4, m_idx * 4 + k_idx % 4] = sf
    elif config.use_tma_aligned_col_major_sf:
        tensor[k_idx, m_idx] = sf
    else:
        tensor[m_idx, k_idx] = sf


# ---------------------------------------------------------------------------
# JIT kernel: swiglu_forward_and_per_token_cast
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _get_swiglu_forward_and_per_token_cast_kernel(
    hidden: int,
    with_weight: bool,
    with_pos_to_expert: bool,
    use_clamp: bool,
    count_clamp: bool,
    in_dtype: T.dtype,
    out_config: _CastOutputConfig,
    num_sms: Optional[int],
):
    num_elems_per_block = 4096
    num_threads = 256
    _, num_per_channels = out_config.sf_block

    TILE_X = 1
    TILE_Y = num_per_channels

    while TILE_Y * 2 <= num_elems_per_block and hidden % (TILE_Y * 2) == 0:
        TILE_Y *= 2

    while TILE_X * TILE_Y % num_threads != 0:
        TILE_X *= 2

    if TILE_X != 1 or TILE_Y < 2048:
        if TILE_X == 1 and hidden <= 8192:
            TILE_Y = hidden

        if _is_power_of_two(TILE_Y):
            while TILE_X * TILE_Y * 2 <= num_elems_per_block:
                TILE_X *= 2

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    num_tokens = T.dynamic('num_tokens')
    num_topk = T.dynamic('num_topk')
    sf_shape = _get_sf_shape((num_expanded_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    num_blocks = T.ceildiv(num_expanded_tokens, TILE_X) * T.ceildiv(hidden, TILE_Y)
    if count_clamp:
        num_blocks = num_sms * 4

    num_groups = TILE_Y // num_per_channels

    @T.prim_func
    def swiglu_forward_and_per_token_cast_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_expanded_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        pos_to_token_topk: T.Tensor[(num_expanded_tokens,), T.int32],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        pos_to_expert: T.Tensor[(num_expanded_tokens,), T.int32],
        clamped_count: T.Tensor[(3,), T.int64],
        swiglu_clamp_value: T.float32,
    ):
        with T.Kernel(num_blocks, threads=num_threads) as pid:
            tid = T.get_thread_binding()

            topk_weights_1d = T.reshape(topk_weights, (num_tokens * num_topk,))
            x_fragment = T.alloc_fragment((TILE_X, TILE_Y), T.float32)
            x_fragment_reshaped = T.reshape(x_fragment, [TILE_X, num_groups, num_per_channels])
            xl_fragment = T.alloc_fragment((TILE_X, TILE_Y), in_dtype)
            xr_fragment = T.alloc_fragment((TILE_X, TILE_Y), in_dtype)

            count_silu = T.alloc_reducer((1,), T.int64, 'sum', replication='all')
            count_upper = T.alloc_reducer((1,), T.int64, 'sum', replication='all')
            count_lower = T.alloc_reducer((1,), T.int64, 'sum', replication='all')

            T.fill(count_silu, 0)
            T.fill(count_upper, 0)
            T.fill(count_lower, 0)

            if count_clamp:
                upper = T.ceildiv(T.ceildiv(num_expanded_tokens, TILE_X) * T.ceildiv(hidden, TILE_Y) - pid, num_blocks)
            else:
                upper = 1

            for iter in T.serial(upper):
                pid_iter = iter * num_blocks + pid
                pid_x, pid_y = pid_iter // T.ceildiv(hidden, TILE_Y), pid_iter % T.ceildiv(hidden, TILE_Y)

                topk_weights_fragment = T.alloc_fragment((TILE_X,), T.float32)
                pos_to_expert_fragment = T.alloc_fragment((TILE_X,), T.int32)
                sf_inv_fragment = T.alloc_fragment((TILE_X, num_groups), T.float32)
                out_fragment = T.alloc_fragment((TILE_X, TILE_Y), out_config.dtype)

                if with_weight:
                    for i in T.Parallel(TILE_X):
                        pos = pos_to_token_topk[pid_x * TILE_X + i]
                        if pos >= 0:
                            T.assume(pos < num_tokens * num_topk)
                            topk_weights_fragment[i] = topk_weights_1d[pos]

                if with_pos_to_expert:
                    for i in T.Parallel(TILE_X):
                        pos_to_expert_fragment[i] = pos_to_expert[pid_x * TILE_X + i]

                if not with_pos_to_expert or TILE_X != 1 or pos_to_expert_fragment[0] >= 0:
                    for i, j in T.Parallel(TILE_X, TILE_Y):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            xl_fragment[i, j] = x[pid_x * TILE_X + i, pid_y * TILE_Y + j]
                            xr_fragment[i, j] = x[pid_x * TILE_X + i, pid_y * TILE_Y + j + hidden]

                    for i, j in T.Parallel(TILE_X, TILE_Y):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            val_l = T.alloc_var(T.float32)
                            val_r = T.alloc_var(T.float32)
                            val_l = T.float32(xl_fragment[i, j])
                            val_r = T.float32(xr_fragment[i, j])
                            if use_clamp:
                                if count_clamp:
                                    clamp_silu = val_l > swiglu_clamp_value
                                    val_l = T.Select(clamp_silu, swiglu_clamp_value, val_l)
                                    count_silu[0] += clamp_silu
                                    clamp_upper = val_r > swiglu_clamp_value
                                    clamp_lower = val_r < -swiglu_clamp_value
                                    val_r = T.Select(clamp_upper, swiglu_clamp_value, val_r)
                                    val_r = T.Select(clamp_lower, -swiglu_clamp_value, val_r)
                                    count_upper[0] += clamp_upper
                                    count_lower[0] += clamp_lower
                                else:
                                    val_l = T.min(val_l, swiglu_clamp_value)
                                    val_r = T.max(T.min(val_r, swiglu_clamp_value), -swiglu_clamp_value)
                            if with_weight:
                                val = val_l / (1 + T.exp(-val_l)) * val_r * topk_weights_fragment[i]
                            else:
                                val = val_l / (1 + T.exp(-val_l)) * val_r
                            x_fragment[i, j] = val

                    T.reduce_absmax(x_fragment_reshaped, sf_inv_fragment, dim=2)
                    for i, j in T.Parallel(TILE_X, num_groups):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            sf, sf_inv = _get_sf_and_inv(sf_inv_fragment[i, j], out_config)
                            x_idx = pid_x * TILE_X + i
                            y_idx = pid_y * num_groups + j
                            _store_sf(out_sf, sf, x_idx, y_idx, out_config)
                            sf_inv_fragment[i, j] = sf_inv

                    for i, j in T.Parallel(TILE_X, TILE_Y):
                        if (not with_pos_to_expert) or pos_to_expert_fragment[i] >= 0:
                            out_fragment[i, j] = x_fragment[i, j] * sf_inv_fragment[i, j // num_per_channels]
                    T.copy(out_fragment, out[pid_x * TILE_X, pid_y * TILE_Y])

            if count_clamp:
                T.finalize_reducer(count_silu)
                T.finalize_reducer(count_upper)
                T.finalize_reducer(count_lower)

                if tid == 0:
                    T.atomic_add(clamped_count[0], count_silu[0])
                    T.atomic_add(clamped_count[1], count_upper[0])
                    T.atomic_add(clamped_count[2], count_lower[0])

    return swiglu_forward_and_per_token_cast_kernel


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    pos_to_expert: Optional[torch.Tensor] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
    sf_clamp_min: Optional[float] = None,
) -> QuantTensor:
    """Fuse SwiGLU forward pass with per-token FP8 quantization.

    Args:
        x: Input 2D contiguous tensor of shape (num_expanded_tokens, hidden * 2).
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_channels: Number of channels in each scaling block (128 or hidden).
        swiglu_clamp_value: Optional clamp threshold for SwiGLU activations.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and sf-factor tensor.
    """
    assert x.dim() == 2 and x.is_contiguous()
    num_expanded_tokens, hidden = x.shape
    hidden = hidden // 2

    assert hidden % 128 == 0
    assert num_per_channels == 128 or num_per_channels == hidden
    assert fmt == 'e4m3'

    out_config = _get_cast_output_config(
        fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf,
        use_packed_ue8m0, custom_clamp_min_value=sf_clamp_min,
    )
    kernel = _get_swiglu_forward_and_per_token_cast_kernel(
        hidden,
        pos_to_token_topk is not None,
        pos_to_expert is not None,
        swiglu_clamp_value is not None,
        clamped_count is not None,
        in_dtype=T.dtype(x.dtype),
        out_config=out_config,
        num_sms=_get_num_sms() if clamped_count is not None else None,
    )

    out = torch.empty((num_expanded_tokens, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    out_sf = _alloc_scaling_factors((num_expanded_tokens, hidden), out_config)
    swiglu_clamp_value = 0 if swiglu_clamp_value is None else swiglu_clamp_value
    if num_expanded_tokens > 0:
        kernel(x, out, out_sf, pos_to_token_topk, topk_weights, pos_to_expert,
               clamped_count, swiglu_clamp_value)

    out_sf = _cast_epilogue(out_sf, num_expanded_tokens, hidden, out_config)
    return out, out_sf


# ---------------------------------------------------------------------------
# Custom op registration (torch.ops.vllm.swiglu_fp8_quant)
# ---------------------------------------------------------------------------

def _swiglu_fp8_quant_fake(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    pos_to_token_topk: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    pos_to_expert: Optional[torch.Tensor] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    swiglu_clamp_value: Optional[float] = None,
    clamped_count: Optional[torch.Tensor] = None,
    sf_clamp_min: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_expanded_tokens = x.size(0)
    hidden = x.size(1) // 2
    out = torch.empty(
        num_expanded_tokens, hidden,
        dtype=torch.float8_e4m3fn, device=x.device,
    )
    out_sf = torch.empty(
        num_expanded_tokens, _ceil_div(hidden, num_per_channels),
        dtype=torch.float32, device=x.device,
    )
    return out, out_sf


from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    op_name="swiglu_fp8_quant",
    op_func=swiglu_forward_and_per_token_cast,
    mutates_args=[],
    fake_impl=_swiglu_fp8_quant_fake,
)
