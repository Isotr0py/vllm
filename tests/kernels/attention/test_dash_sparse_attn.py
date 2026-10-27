# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DASH bias-aware block-sparse attention parity against a PyTorch reference.

The reference gathers the K/V of the selected remote blocks out of the paged
cache, adds the per-(query, slot) bias to their logits, unions them with the
unbiased causal local window, and runs an fp32 softmax -- the same contract
the Triton kernels implement.
"""

import pytest
import torch

from vllm.model_executor.kernels.attention.dash.attend_triton import (
    dash_sparse_attn,
    dash_sparse_attn_decode,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="DASH kernels are CUDA-only"
    ),
]

SPARSE_BLOCK_SIZE = 32
LOCAL_BLOCKS = 4  # 128-token local window, as in Unlimited-OCR-2
TOPK = 12
HEAD_DIM = 128


def _ref_dash_attention(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_cache: torch.Tensor,  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    blk_bias: torch.Tensor | None,
    block_table: torch.Tensor,  # [batch, max_blocks]
    req_ids: torch.Tensor,  # [total_q]
    query_positions: torch.Tensor,  # [total_q] request-local positions
    sm_scale: float,
) -> torch.Tensor:
    total_q, num_heads, _ = q.shape
    num_kv_heads = k_cache.shape[2]
    group = num_heads // num_kv_heads
    lw = LOCAL_BLOCKS * SPARSE_BLOCK_SIZE
    out = torch.zeros(
        total_q, num_heads, HEAD_DIM, dtype=torch.float32, device=q.device
    )
    for req in range(block_table.shape[0]):
        rows = (req_ids == req).nonzero(as_tuple=True)[0]
        t = query_positions[rows]  # [R]
        pos = torch.arange(int(t.max()) + 1, device=q.device)  # [P]
        pages = block_table[req]
        page_ids = pages[pos // SPARSE_BLOCK_SIZE].long()
        k = k_cache[page_ids, pos % SPARSE_BLOCK_SIZE].float()  # [P, Hkv, D]
        v = v_cache[page_ids, pos % SPARSE_BLOCK_SIZE].float()
        q_r = q[rows].float().view(-1, num_kv_heads, group, HEAD_DIM)
        logits = torch.einsum("rkgd,pkd->rkgp", q_r, k) * sm_scale

        nblocks = pos[-1].item() // SPARSE_BLOCK_SIZE + 1
        sel = topk_idx[:, rows].long()  # [Hkv, R, topk]
        # Route -1 (padding) slots to an overflow column, then drop it;
        # scattering them to block 0 would clobber real selections.
        sel = sel.clamp(min=0).masked_fill(sel < 0, nblocks)
        block_sel = torch.zeros(
            num_kv_heads,
            rows.numel(),
            nblocks + 1,
            dtype=torch.bool,
            device=q.device,
        ).scatter_(-1, sel, True)[..., :nblocks]
        if blk_bias is None:
            bias_dense = torch.zeros(
                num_kv_heads,
                rows.numel(),
                nblocks,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            bias_dense = torch.zeros(
                num_kv_heads,
                rows.numel(),
                nblocks + 1,
                dtype=torch.float32,
                device=q.device,
            ).scatter_(-1, sel, blk_bias[:, rows])[..., :nblocks]
        blk_of_pos = pos // SPARSE_BLOCK_SIZE
        sel_mask = block_sel[:, :, blk_of_pos]  # [Hkv, R, P]
        bias_vec = bias_dense[:, :, blk_of_pos]  # [Hkv, R, P]

        local_start = (t - lw + 1).clamp(min=0)  # [R]
        causal = pos[None, :] <= t[:, None]  # [R, P]
        is_local = pos[None, :] >= local_start[:, None]  # [R, P]
        remote = sel_mask & ~is_local[None] & causal[None]  # [Hkv, R, P]
        keep = causal[None] & (is_local[None] | remote)
        logits = logits + torch.where(
            remote.permute(1, 0, 2)[:, :, None, :],
            bias_vec.permute(1, 0, 2)[:, :, None, :],
            torch.zeros((), device=q.device),
        )
        logits = logits.masked_fill(
            ~keep.permute(1, 0, 2)[:, :, None, :], float("-inf")
        )
        p = torch.softmax(logits, dim=-1)
        out[rows] = torch.einsum("rkgp,pkd->rkgd", p, v).reshape(
            -1, num_heads, HEAD_DIM
        )
    return out


def _make_case(
    query_lens: list[int],
    context_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
):
    device = torch.device("cuda:0")
    gen_cpu = torch.Generator().manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(query_lens)
    seq_lens_list = [qlen + ctx for qlen, ctx in zip(query_lens, context_lens)]
    total_q = sum(query_lens)
    blocks_per_seq = [triton.cdiv(seq, SPARSE_BLOCK_SIZE) for seq in seq_lens_list]
    num_pages = sum(blocks_per_seq)
    q = torch.randn(
        total_q, num_heads, HEAD_DIM, dtype=dtype, device=device, generator=gen
    )
    k_cache = torch.randn(
        num_pages,
        SPARSE_BLOCK_SIZE,
        num_kv_heads,
        HEAD_DIM,
        dtype=dtype,
        device=device,
        generator=gen,
    )
    v_cache = torch.randn_like(k_cache)
    perm = torch.randperm(num_pages, generator=gen_cpu, dtype=torch.int32)
    max_blocks = max(blocks_per_seq)
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32)
    offset = 0
    for req, nblk in enumerate(blocks_per_seq):
        block_table[req, :nblk] = perm[offset : offset + nblk]
        offset += nblk
    block_table = block_table.to(device)

    req_ids = torch.cat(
        [torch.full((qlen,), req) for req, qlen in enumerate(query_lens)]
    )
    query_positions = torch.cat(
        [torch.arange(ctx, ctx + qlen) for qlen, ctx in zip(query_lens, context_lens)]
    )
    # Per-(kv head, row) unique remote blocks: any block with at least one key
    # strictly below the local window, padded with -1.
    lw = LOCAL_BLOCKS * SPARSE_BLOCK_SIZE
    topk_idx = torch.full((num_kv_heads, total_q, TOPK), -1, dtype=torch.int32)
    for row in range(total_q):
        t = int(query_positions[row])
        local_start = max(0, t - lw + 1)
        n_eligible = triton.cdiv(local_start, SPARSE_BLOCK_SIZE)
        n = min(TOPK, n_eligible)
        if n == 0:
            continue
        for kh in range(num_kv_heads):
            sample = torch.randperm(n_eligible, generator=gen_cpu)[:n]
            topk_idx[kh, row, :n] = sample.to(torch.int32)
    blk_bias = torch.randn(num_kv_heads, total_q, TOPK, generator=gen_cpu) * 0.5
    return (
        q,
        k_cache,
        v_cache,
        topk_idx.to(device),
        blk_bias.float().to(device),
        block_table,
        req_ids.to(device),
        query_positions.to(device),
        torch.tensor(seq_lens_list, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize(("num_heads", "num_kv_heads"), [(10, 10), (8, 2)])
@pytest.mark.parametrize(
    ("query_lens", "context_lens"),
    [
        ([1, 37, 200, 513], [0, 0, 0, 0]),
        ([16, 64], [300, 128]),  # chunked prefill with a cached prefix
        ([1, 3], [0, 2]),  # no remote candidates: local window only
    ],
)
def test_dash_sparse_attn_prefill_parity(
    query_lens, context_lens, num_heads, num_kv_heads
):
    (
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        req_ids,
        query_positions,
        seq_lens,
    ) = _make_case(query_lens, context_lens, num_heads, num_kv_heads, seed=0)
    sm_scale = HEAD_DIM**-0.5
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=q.device,
    )
    prefix_lens = torch.tensor(context_lens, dtype=torch.int32, device=q.device)
    output = torch.empty_like(q)

    dash_sparse_attn(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max(query_lens),
        num_kv_heads,
        sm_scale,
        output,
        local_blocks=LOCAL_BLOCKS,
        sparse_block_size=SPARSE_BLOCK_SIZE,
    )

    ref = _ref_dash_attention(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        req_ids,
        query_positions,
        sm_scale,
    )
    torch.testing.assert_close(output.float(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(("num_heads", "num_kv_heads"), [(10, 10), (8, 2)])
@pytest.mark.parametrize("decode_query_len", [1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
def test_dash_sparse_attn_decode_parity(
    num_heads, num_kv_heads, decode_query_len, with_bias
):
    seq_lens_list = [2, 33, 160, 1024]
    batch = len(seq_lens_list)
    (
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        req_ids,
        query_positions,
        seq_lens,
    ) = _make_case(
        [decode_query_len] * batch,
        [seq - decode_query_len for seq in seq_lens_list],
        num_heads,
        num_kv_heads,
        seed=1,
    )
    sm_scale = HEAD_DIM**-0.5
    if not with_bias:
        blk_bias = None
    output = torch.empty_like(q)

    dash_sparse_attn_decode(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
        local_blocks=LOCAL_BLOCKS,
        sparse_block_size=SPARSE_BLOCK_SIZE,
    )

    ref = _ref_dash_attention(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        req_ids,
        query_positions,
        sm_scale,
    )
    torch.testing.assert_close(output.float(), ref, atol=1e-2, rtol=1e-2)
