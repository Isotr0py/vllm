# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DASH SR-SWA attention layer for baidu/Unlimited-OCR-2.

Replaces ``DeepseekAttention`` in every decoder layer when the checkpoint
carries ``sparse_attention_config`` (the v2 marker). The layer:

1. projects q/k/v as the plain-MHA DeepseekAttention does;
2. applies the page2d two-axis RoPE to q/k (blended per token with the 1D
   YaRN fallback for requests whose prompt has no ``<PAGE>`` markers);
3. writes k/v to the standard paged KV cache (``FullAttentionSpec``,
   ``[num_blocks, H_kv, 32, 2*head_dim]`` logical view) and the router index
   keys to a per-layer fp32 index cache (``[num_blocks, 1, 32, index_dim]``);
4. routes: ``head_cls`` block summaries over the full cached index keys ->
   block scores -> page-aware same-page candidate mask -> top-M entmax
   window + hard top-K read set with a mean-centered log-prior bias
   (``DashRouter``);
5. launches the bias-aware block-sparse Triton kernel
   (:func:`dash_sparse_attn`), which attends the selected remote blocks plus
   the unbiased 128-token local window in a single online softmax.

The per-token page coordinates arrive as extra position channels prepared by
the runner-side ``Page2dRopeState`` (see ``vllm/v1/worker/gpu/mm/page_rope.py``):
channel 0 = flat 1D position, 1 = intra-page token offset, 2 = page id
(-1 = page2d inactive, use 1D RoPE). The per-KV-token image page ids used
for the candidate mask arrive via :class:`DashContext` on the model.

Eager-only for now (per-request host loop in ``_route`` and a ``tolist()``
sync on seq_lens); CUDA-graph support is future work.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import extract_layer_index
from vllm.transformers_utils.configs.unlimited_ocr import (
    UnlimitedOCRSparseAttentionConfig,
)
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from .backend import DashAttentionBackend, DashIndexBackend
from .dash_router import DashRouter, GemmaRMSNorm, apply_rotary_pos_emb
from .page2d_rope import Page2DRotaryEmbedding

# Extra position channels prepared by the runner-side Page2dRopeState.
PAGE_ROPE_NUM_DIMS = 4
PAGE_ROPE_FLAT_CH = 0
PAGE_ROPE_TOK_CH = 1
PAGE_ROPE_PAGE_CH = 2
PAGE_ROPE_IMG_CH = 3


@dataclass
class DashContext:
    """Per-batch page-routing context handed to the model by the runner.

    Attributes:
        img_page_tok: [B, S_max] int32 image page id per KV token (-1 for
            non-image and generated tokens), gathered from the runner's
            staged prefill channels.

    """

    img_page_tok: torch.Tensor | None


class DashContextProvider:
    """Plain (non-module) holder shared by the model and its DASH layers.

    Storing the top-level model on a submodule would register it as a
    submodule of itself (nn.Module.__setattr__) and recurse; this holder is a
    plain object, so it never enters the module tree.

    Attributes:
        dash_ctx: the current batch's page-routing context.
        positions: the full [4, T] page2d position channels of the current
            batch (the language model itself only receives the flat channel,
            since DeepseekV2Model assumes 1-D positions).

    """

    def __init__(self) -> None:
        self.dash_ctx: DashContext | None = None
        self.positions: torch.Tensor | None = None


class DashAttention(Attention):
    """Attention layer marker for DASH layers.

    Registers the layer for KV-cache allocation with the pass-through
    :class:`DashAttentionBackend`. ``forward`` is never called; the owning
    :class:`UnlimitedOCR2Attention` writes the cache and launches the DASH
    kernels itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["attn_backend"] = DashAttentionBackend
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "DashAttention.forward must never run: UnlimitedOCR2Attention "
            "drives the DASH kernels directly."
        )


class DashIndexCache(nn.Module, AttentionLayerBase):
    """Per-layer fp32 router index-key cache ([num_blocks, 1, 32, index_dim]).

    Same paged layout and block size as the main KV cache, but one head and
    K-only content (``head_size_v=0``), so each page holds one 32-token sparse
    block of fp32 index keys. It is its own KV cache group (the spec differs
    from the main KV cache), hence its own block table; both are read from
    the forward context at attention time.
    """

    def __init__(self, index_dim: int, prefix: str, cache_config) -> None:
        super().__init__()
        from vllm.config import get_current_vllm_config

        self.kv_cache = torch.tensor([])
        self.index_dim = index_dim
        self.prefix = prefix
        self.cache_config = cache_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.index_dim,
            head_size_v=0,
            dtype=torch.float32,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DashIndexBackend

    def forward(self): ...


class _DashIndexerWeights(nn.Module):
    """Checkpoint-parity holder for ``self_attn.indexer.*`` weights.

    The HF reference keeps the page/rope helpers and a 1D/2D rope on the
    indexer module; in this port those live in
    :mod:`vllm.model_executor.models.unlimited_ocr2.page2d_rope` and the
    released checkpoint (``training_stage == "dash"``) scores blocks with the
    ``dash_router`` weights only. The indexer projections are therefore
    loaded (to align with the checkpoint) but never run.
    """

    def __init__(self, hidden_size: int, index_dim: int, rms_norm_eps: float):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.q_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)


class UnlimitedOCR2Attention(nn.Module):
    """DASH SR-SWA attention (drop-in replacement for DeepseekAttention)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 8192,
        cache_config=None,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        sp: UnlimitedOCRSparseAttentionConfig | None = getattr(
            config, "sparse_attention_config", None
        )
        if sp is None:
            raise ValueError("UnlimitedOCR2Attention requires sparse_attention_config")
        self.sp = sp
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.index_dim = sp.sparse_index_dim
        self.sparse_block_size = sp.sparse_block_size
        self.local_blocks = max(1, sp.sparse_local_block)
        self.use_page2d = sp.sparse_main_rope_mode == "page2d"
        self.router_page2d = sp.sparse_rope_mode == "page2d"
        # With index_dim == head_dim the router shares the main rope's 2-axis
        # cos/sin; otherwise it falls back to 1D like the reference does.
        self.router_shares_rope = self.router_page2d and self.index_dim == self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
        )

        # 1D YaRN fallback (text-only requests): cos/sin come straight from
        # the vLLM rope cache, which is [cos(half) | sin(half)] per position.
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.rope2d = (
            Page2DRotaryEmbedding(
                dim=self.head_dim,
                base_tok=sp.sparse_rope_tok_theta,
                base_page=sp.sparse_rope_page_theta,
                page_pairs=sp.sparse_rope_page_pairs,
                tok_freq_mode=sp.sparse_rope_tok_freq_mode,
                page_mod=sp.sparse_rope_page_mod,
            )
            if self.use_page2d
            else None
        )

        rms_eps = getattr(config, "rms_norm_eps", None) or 1e-6
        self.dash_router = DashRouter(
            hidden_size=hidden_size,
            index_dim=self.index_dim,
            block_size=self.sparse_block_size,
            local_blocks=self.local_blocks,
            sigma=sp.dash_sigma,
            rms_norm_eps=rms_eps,
            stage0=sp.dash_stage0,
            q_chunk=sp.dash_query_chunk,
        )
        layer_idx = extract_layer_index(prefix)
        freq = sp.sparse_attention_freq
        slot = sum(1 for f in freq[:layer_idx] if f) if freq else layer_idx

        def _per_layer(arr, default):
            return arr[slot] if arr and slot < len(arr) else default

        self.dash_router.set_inference_params(
            idx_scale=_per_layer(sp.dash_scale_per_layer, sp.dash_scaling_factor)
            / (self.index_dim**0.5),
            floor=_per_layer(sp.dash_floor_per_layer, sp.dash_prior_floor),
            topk=sp.dash_topk_blocks,
            topm=sp.sparse_topk_blocks,
            sigma=sp.dash_sigma,
        )
        self.indexer = _DashIndexerWeights(hidden_size, self.index_dim, rms_eps)

        self.attn = DashAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.idx_cache = DashIndexCache(
            index_dim=self.index_dim,
            prefix=f"{prefix}.idx_cache",
            cache_config=cache_config,
        )
        # Wired by UnlimitedOCRForCausalLM after construction; provides the
        # per-batch DashContext.
        self.dash_ctx_provider: Any = None

    # -- rope -----------------------------------------------------------------

    def _cos_sin_1d(self, flat_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """1D YaRN cos/sin [T, D] in transformers (rotate-half) layout."""
        cs = self.rotary_emb.cos_sin_cache[flat_pos.clamp(min=0)].float()
        half = cs.shape[-1] // 2
        cos = torch.cat((cs[:, :half], cs[:, :half]), dim=-1)
        sin = torch.cat((cs[:, half:], cs[:, half:]), dim=-1)
        return cos, sin

    def _blended_cos_sin(
        self,
        flat_pos: torch.Tensor,
        tok_pos: torch.Tensor,
        page_pos: torch.Tensor,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token blend of page2d and 1D YaRN cos/sin.

        Tokens with ``page_pos < 0`` belong to requests whose prompt has no
        page markers; they use the 1D YaRN fallback exactly like the
        reference's 1D branch.
        """
        cos, sin = self._cos_sin_1d(flat_pos)
        if self.rope2d is not None:
            cos2, sin2 = self.rope2d(
                ref,
                tok_pos.clamp(min=0).unsqueeze(0),
                page_pos.clamp(min=0).unsqueeze(0),
            )
            use2d = (page_pos >= 0).unsqueeze(-1)
            cos = torch.where(use2d, cos2[0].float(), cos)
            sin = torch.where(use2d, sin2[0].float(), sin)
        return cos.to(ref.dtype), sin.to(ref.dtype)

    # -- caches ----------------------------------------------------------------

    def _write_caches(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_k: torch.Tensor,
        md: CommonAttentionMetadata,
        md_idx: CommonAttentionMetadata,
    ) -> None:
        bs = self.sparse_block_size
        slot = md.slot_mapping.long()
        kv = self.attn.kv_cache  # [num_blocks, H_kv, bs, 2*D]
        kv[slot // bs, :, slot % bs] = torch.cat((k, v), dim=-1).to(kv.dtype)

        slot_i = md_idx.slot_mapping.long()
        idx_kv = self.idx_cache.kv_cache  # [num_blocks, 1, bs, index_dim]
        idx_kv[slot_i // bs, 0, slot_i % bs] = idx_k.to(idx_kv.dtype)

    def _kernel_kv_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """[num_blocks, 32, H_kv, D] K and V views for the DASH kernel."""
        kv_t = self.attn.kv_cache.transpose(1, 2)
        return kv_t.split(self.head_dim, dim=-1)

    # -- routing ---------------------------------------------------------------

    def _candidate_mask(
        self,
        query_page: torch.Tensor,
        img_page_tok: torch.Tensor,
        seq_len: int,
        nblk: int,
    ) -> torch.Tensor:
        """Same-page image-block candidates [1, L_q, nblk] for one request."""
        bs = self.sparse_block_size
        img_blk = F.pad(img_page_tok[:seq_len], (0, nblk * bs - seq_len), value=-1)
        img_blk = img_blk.view(nblk, bs)
        qp = query_page[:, None, None]
        same = (qp >= 0) & (qp == img_blk.unsqueeze(0))
        return same.any(dim=-1).unsqueeze(0)

    def _route(
        self,
        idx_q: torch.Tensor,
        flat_pos: torch.Tensor,
        page_pos: torch.Tensor,
        md: CommonAttentionMetadata,
        md_idx: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route every query token to its top-K remote blocks.

        Returns (topk_idx [1, T, K] int32 with -1 padding, blk_bias [1, T, K]
        fp32), broadcast across KV heads by the caller.
        """
        # NOTE(eager-only): per-request host loop with a seq_lens sync.
        # TODO(dash): batch the gather/scoring and drop the sync for CUDA
        # graph support.
        T = idx_q.shape[0]
        device = idx_q.device
        topk = self.dash_router.route_params.topk
        topk_idx = torch.full((1, T, topk), -1, dtype=torch.int32, device=device)
        blk_bias = torch.zeros((1, T, topk), dtype=torch.float32, device=device)

        qsl = md.query_start_loc_cpu.tolist()
        seq_lens = md.seq_lens.tolist()
        ctx: DashContext | None = (
            self.dash_ctx_provider.dash_ctx if self.dash_ctx_provider else None
        )
        img_page_tok = ctx.img_page_tok if ctx is not None else None

        bs = self.sparse_block_size
        idx_kv = self.idx_cache.kv_cache
        idx_bt = md_idx.block_table_tensor
        for r, seq_len in enumerate(seq_lens):
            q0, q1 = qsl[r], qsl[r + 1]
            if q1 <= q0 or seq_len <= 0:
                continue
            nblk = -(-seq_len // bs)
            pages = idx_bt[r, :nblk].long()
            idx_k_full = idx_kv[pages, 0].reshape(nblk * bs, -1)[:seq_len]

            k_bar, chunk_valid = self.dash_router.head_cls_summary(
                idx_k_full.unsqueeze(0), nblk
            )
            scores = self.dash_router.block_scores(
                idx_q[q0:q1].unsqueeze(0),
                flat_pos[q0:q1].unsqueeze(0),
                nblk,
                k_bar=k_bar,
                chunk_valid=chunk_valid,
            )
            cand = None
            if img_page_tok is not None:
                cand = self._candidate_mask(
                    page_pos[q0:q1] - 1, img_page_tok[r], seq_len, nblk
                )
            topi, bias, valid = self.dash_router.route_and_select(scores, cand)
            # K' = min(topk, topm, nblk) columns are routed; the rest of the
            # fixed-width buffer stays -1 padding for the kernel.
            k_sel = topi.shape[-1]
            topk_idx[0, q0:q1, :k_sel] = torch.where(
                valid[0], topi[0].to(torch.int32), topk_idx.new_full((), -1)
            )
            blk_bias[0, q0:q1, :k_sel] = bias[0]
        return topk_idx, blk_bias

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]
        # ``positions`` here is the flat channel (DeepseekV2Model requires
        # 1-D positions); the page2d channels travel via the provider.
        if (
            self.dash_ctx_provider is not None
            and self.dash_ctx_provider.positions is not None
        ):
            channels = self.dash_ctx_provider.positions
            tok_pos = channels[PAGE_ROPE_TOK_CH, :T]
            page_pos = channels[PAGE_ROPE_PAGE_CH, :T]
        else:
            # No provider (unit tests / unwired layer): page2d inactive.
            tok_pos = positions
            page_pos = torch.full_like(positions, -1)
        flat_pos = positions

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(T, self.num_heads, self.head_dim)
        k = k.view(T, self.num_kv_heads, self.head_dim)
        v = v.view(T, self.num_kv_heads, self.head_dim)

        cos, sin = self._blended_cos_sin(flat_pos, tok_pos, page_pos, v)
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        q4, k4 = apply_rotary_pos_emb(q4, k4, cos.unsqueeze(0), sin.unsqueeze(0))
        q = q4.squeeze(0).transpose(0, 1).contiguous()
        k = k4.squeeze(0).transpose(0, 1)

        router_pe = (
            (cos.unsqueeze(0), sin.unsqueeze(0)) if self.router_shares_rope else None
        )
        if router_pe is None:
            cos1, sin1 = self._cos_sin_1d(flat_pos)
            router_pe = (cos1.unsqueeze(0).to(v.dtype), sin1.unsqueeze(0).to(v.dtype))
        fwd_ctx = get_forward_context()
        attn_md = fwd_ctx.attn_metadata
        if attn_md is None or self.attn.layer_name not in attn_md:
            # Initial memory profiling runs the model without attention
            # metadata; only the projections matter there.
            attn_output, _ = self.o_proj(q.reshape(T, self.num_heads * self.head_dim))
            return attn_output

        idx_q, idx_k = self.dash_router.index_qk(hidden_states.unsqueeze(0), router_pe)
        idx_q, idx_k = idx_q[0], idx_k[0]

        attn_map = cast(dict[str, CommonAttentionMetadata], attn_md)
        md = attn_map[self.attn.layer_name]
        md_idx = attn_map[self.idx_cache.prefix]
        self._write_caches(k, v, idx_k, md, md_idx)

        topk_idx, blk_bias = self._route(idx_q, flat_pos, page_pos, md, md_idx)

        output = torch.empty_like(q)
        k_cache, v_cache = self._kernel_kv_views()
        cu_seqlens_q = md.query_start_loc.to(torch.int32)
        seq_lens = md.seq_lens.to(torch.int32)
        query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        prefix_lens = seq_lens - query_lens

        from vllm.model_executor.kernels.attention.dash.attend_triton import (
            dash_sparse_attn,
        )

        dash_sparse_attn(
            q,
            k_cache,
            v_cache,
            topk_idx.expand(self.num_kv_heads, -1, -1),
            blk_bias.expand(self.num_kv_heads, -1, -1),
            md.block_table_tensor,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            md.max_query_len,
            self.num_kv_heads,
            self.scaling,
            output,
            local_blocks=self.local_blocks,
            sparse_block_size=self.sparse_block_size,
        )
        attn_output = output.view(T, self.num_heads * self.head_dim)
        attn_output, _ = self.o_proj(attn_output)
        return attn_output
