# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU integration tests for the Unlimited-OCR-2 DASH forward path.

Covers the two new pieces of glue whose math is not already pinned by the
parity suite (``test_dash_page2d_parity.py``):

* ``Page2dRopeState`` -- the runner-side page2d position channels and the
  decode state machine (prefill seeding, chunked prefill slices, decode
  advance, mixed prefill+decode batches, the page-routing context).
* ``UnlimitedOCR2Attention`` -- RoPE blending (page2d vs 1D YaRN fallback),
  paged cache writes, and the routing orchestration (index-cache gather,
  page-aware candidate mask, top-K/bias assembly) against independent
  recomputation through the public DashRouter API.
"""

import math

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.models.unlimited_ocr2 import (
    DashRouter,
    PageRopeState,
    build_page_aware_ids,
    build_page_rope_positions,
)
from vllm.model_executor.models.unlimited_ocr2.attention import (
    PAGE_ROPE_FLAT_CH,
    PAGE_ROPE_IMG_CH,
    PAGE_ROPE_PAGE_CH,
    PAGE_ROPE_TOK_CH,
    DashContext,
    UnlimitedOCR2Attention,
)
from vllm.transformers_utils.configs.unlimited_ocr import (
    UnlimitedOCRSparseAttentionConfig,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu.mm.page_rope import Page2dRopeState

IMG_ID = 128815
PAGE_ID = 128827
PER_PAGE = 273
BLOCK = 32


def _page_prompt_ids(num_pages: int, instr_len: int = 3) -> list[int]:
    ids = [1]  # BOS / system prefix (page bucket 0)
    for _ in range(num_pages):
        ids.append(PAGE_ID)
        ids.extend([IMG_ID] * PER_PAGE)
    ids.extend([7] * instr_len)
    return ids


def _make_rope_state(max_num_reqs: int = 2) -> Page2dRopeState:
    return Page2dRopeState(
        image_token_id=IMG_ID,
        page_token_id=PAGE_ID,
        per_page_image_tokens=PER_PAGE,
        max_num_reqs=max_num_reqs,
        max_num_tokens=4096,
        max_model_len=8192,
        device=torch.device("cpu"),
    )


class TestPage2dRopeState:
    def test_prefill_channels_match_reference_builders(self):
        ids = _page_prompt_ids(2)
        seq_len = len(ids)
        state = _make_rope_state()
        state.init_prefill_positions(0, None, ids, [])
        state.apply_staged_writes()

        qsl = torch.tensor([0, seq_len])
        state.prepare_positions(
            idx_mapping=torch.tensor([0]),
            query_start_loc=qsl,
            prefill_lens=torch.tensor([seq_len, 0]),
            num_computed_tokens=torch.zeros(2, dtype=torch.long),
            input_ids=torch.tensor(ids, dtype=torch.int32),
            num_tokens=seq_len,
        )
        positions = state.get_positions(seq_len)
        assert torch.equal(positions[PAGE_ROPE_FLAT_CH], torch.arange(seq_len))

        tok, page, pf_cnt, n_pg = build_page_rope_positions(
            torch.tensor(ids).unsqueeze(0), IMG_ID, PAGE_ID
        )
        assert torch.equal(positions[PAGE_ROPE_TOK_CH], tok[0])
        assert torch.equal(positions[PAGE_ROPE_PAGE_CH], page[0])

        _, img_ids, _ = build_page_aware_ids(
            torch.tensor(ids).unsqueeze(0), IMG_ID, PAGE_ID, PER_PAGE
        )
        assert torch.equal(positions[PAGE_ROPE_IMG_CH], img_ids[0].long())

        # The decode state is seeded from the last prefill token.
        assert state.dec_page[0].item() == int(page[0, -1])
        assert state.dec_tok[0].item() == int(tok[0, -1])
        assert state.dec_npg[0].item() == n_pg
        assert torch.equal(state.dec_pfcnt[0, :n_pg].long(), pf_cnt[0])

    def test_chunked_prefill_reads_staged_slices(self):
        ids = _page_prompt_ids(2)
        seq_len = len(ids)
        state = _make_rope_state()
        state.init_prefill_positions(0, None, ids, [])
        state.apply_staged_writes()

        chunk = slice(100, 250)
        state.prepare_positions(
            idx_mapping=torch.tensor([0]),
            query_start_loc=torch.tensor([0, chunk.stop - chunk.start]),
            prefill_lens=torch.tensor([seq_len, 0]),
            num_computed_tokens=torch.tensor([chunk.start, 0]),
            input_ids=torch.tensor(ids[chunk], dtype=torch.int32),
            num_tokens=chunk.stop - chunk.start,
        )
        positions = state.get_positions(chunk.stop - chunk.start)
        tok, page, _, _ = build_page_rope_positions(
            torch.tensor(ids).unsqueeze(0), IMG_ID, PAGE_ID
        )
        assert torch.equal(
            positions[PAGE_ROPE_FLAT_CH], torch.arange(chunk.start, chunk.stop)
        )
        assert torch.equal(positions[PAGE_ROPE_TOK_CH], tok[0, chunk])
        assert torch.equal(positions[PAGE_ROPE_PAGE_CH], page[0, chunk])
        # A mid-prefill chunk must not advance the decode state.
        assert state.dec_tok[0].item() == int(tok[0, -1])

    def test_decode_advance_matches_model_side_state_machine(self):
        ids = _page_prompt_ids(2)
        seq_len = len(ids)
        state = _make_rope_state()
        state.init_prefill_positions(0, None, ids, [])
        state.apply_staged_writes()

        tok, page, pf_cnt, n_pg = build_page_rope_positions(
            torch.tensor(ids).unsqueeze(0), IMG_ID, PAGE_ID
        )
        ref = PageRopeState.from_prefill(tok, page, pf_cnt, n_pg)

        # Decode: two text tokens, then a <PAGE> marker, then two more.
        gen = [42, 43, PAGE_ID, 44, 45]
        n_computed = seq_len
        for new_tok in gen:
            state.prepare_positions(
                idx_mapping=torch.tensor([0]),
                query_start_loc=torch.tensor([0, 1]),
                prefill_lens=torch.tensor([seq_len, 0]),
                num_computed_tokens=torch.tensor([n_computed, 0]),
                input_ids=torch.tensor([new_tok], dtype=torch.int32),
                num_tokens=1,
            )
            ref_tok, ref_page = ref.advance(torch.tensor([[new_tok]]), PAGE_ID)
            positions = state.get_positions(1)
            assert positions[PAGE_ROPE_FLAT_CH].item() == n_computed
            assert positions[PAGE_ROPE_TOK_CH].item() == int(ref_tok[0, 0])
            assert positions[PAGE_ROPE_PAGE_CH].item() == int(ref_page[0, 0])
            n_computed += 1

    def test_mixed_prefill_and_decode_batch(self):
        ids0 = _page_prompt_ids(1)
        ids1 = _page_prompt_ids(2)
        state = _make_rope_state()
        state.init_prefill_positions(0, None, ids0, [])
        state.init_prefill_positions(1, None, ids1, [])
        state.apply_staged_writes()

        # req0 decodes (prefill done), req1 is mid-chunked-prefill.
        qsl = torch.tensor([0, 1, 1 + 50])
        state.prepare_positions(
            idx_mapping=torch.tensor([0, 1]),
            query_start_loc=qsl,
            prefill_lens=torch.tensor([len(ids0), len(ids1)]),
            num_computed_tokens=torch.tensor([len(ids0), 10]),
            input_ids=torch.tensor([PAGE_ID] + ids1[10:60], dtype=torch.int32),
            num_tokens=51,
        )
        positions = state.get_positions(51)
        # req0's decode token advanced to page 1 (its first <PAGE>).
        assert positions[PAGE_ROPE_PAGE_CH, 0].item() == 1
        # req1's chunk tokens came from the staged channels.
        tok1, page1, _, _ = build_page_rope_positions(
            torch.tensor(ids1).unsqueeze(0), IMG_ID, PAGE_ID
        )
        assert torch.equal(positions[PAGE_ROPE_TOK_CH, 1:], tok1[0, 10:60])
        assert torch.equal(positions[PAGE_ROPE_PAGE_CH, 1:], page1[0, 10:60])

    def test_dash_context_covers_kv_range(self):
        ids = _page_prompt_ids(2)
        seq_len = len(ids)
        state = _make_rope_state()
        state.init_prefill_positions(0, None, ids, [])
        state.apply_staged_writes()

        # KV range extends past the prompt (generated tokens): image page
        # ids must cover the prompt and pad -1 beyond.
        ctx = state.build_dash_context(torch.tensor([0]), seq_len + 70)
        img = ctx.img_page_tok[0]
        _, img_ids, _ = build_page_aware_ids(
            torch.tensor(ids).unsqueeze(0), IMG_ID, PAGE_ID, PER_PAGE
        )
        assert torch.equal(img[:seq_len].long(), img_ids[0].long())
        assert (img[seq_len:] == -1).all()

    def test_text_only_request_falls_back_to_1d(self):
        ids = [1, 5, 6, 7, 8]
        state = _make_rope_state()
        state.init_prefill_positions(0, None, ids, [])
        state.apply_staged_writes()
        state.prepare_positions(
            idx_mapping=torch.tensor([0]),
            query_start_loc=torch.tensor([0, len(ids)]),
            prefill_lens=torch.tensor([len(ids), 0]),
            num_computed_tokens=torch.zeros(2, dtype=torch.long),
            input_ids=torch.tensor(ids, dtype=torch.int32),
            num_tokens=len(ids),
        )
        positions = state.get_positions(len(ids))
        assert (positions[PAGE_ROPE_PAGE_CH] == -1).all()
        assert torch.equal(positions[PAGE_ROPE_TOK_CH], torch.arange(len(ids)))

    def test_slot_reuse_clears_stale_image_ids(self):
        state = _make_rope_state()
        state.init_prefill_positions(0, None, _page_prompt_ids(3), [])
        state.apply_staged_writes()
        # A shorter text-only request reuses the slot.
        state.init_prefill_positions(0, None, [1, 2, 3], [])
        state.apply_staged_writes()
        ctx = state.build_dash_context(torch.tensor([0]), 800)
        assert (ctx.img_page_tok[0, 3:] == -1).all()


def _make_text_config() -> "object":
    from transformers import DeepseekV2Config

    config = DeepseekV2Config(
        hidden_size=1280,
        num_attention_heads=10,
        num_key_value_heads=10,
        num_hidden_layers=12,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rope_scaling={
            "attention_factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "rope_type": "yarn",
            "type": "yarn",
        },
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=128,
        use_mla=False,
    )
    config.sparse_attention_config = UnlimitedOCRSparseAttentionConfig(
        dash_scale_per_layer=[128.0] * 12,
        dash_floor_per_layer=[1e-4] * 12,
        sparse_attention_freq=[1] * 12,
    )
    return config


@pytest.fixture
def dash_layer(default_vllm_config):
    """Real single-rank DASH attention layer on CPU (gloo TP=1)."""
    import contextlib
    import os
    import tempfile

    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        layer = UnlimitedOCR2Attention(
            vllm_config=default_vllm_config,
            config=_make_text_config(),
            hidden_size=1280,
            num_heads=10,
            max_position_embeddings=131072,
            cache_config=CacheConfig(block_size=BLOCK),
            prefix="model.layers.0.self_attn",
        )
        yield layer
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def _make_metadata(
    query_lens: list[int],
    seq_lens: list[int],
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata:
    cu = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()])
    return CommonAttentionMetadata(
        query_start_loc=cu.to(torch.int32),
        query_start_loc_cpu=cu,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )


def _slot_mapping(
    block_table: torch.Tensor, seq_lens: list[int], context_lens: list[int]
) -> torch.Tensor:
    """Slots (block_id * 32 + offset) of the NEW tokens of each request."""
    slots = []
    for r, (seq_len, ctx) in enumerate(zip(seq_lens, context_lens)):
        for pos in range(ctx, seq_len):
            slots.append(int(block_table[r, pos // BLOCK]) * BLOCK + pos % BLOCK)
    return torch.tensor(slots, dtype=torch.long)


class TestUnlimitedOCR2Attention:
    def test_rope_blend_matches_reference_rotations(self, dash_layer):
        layer = dash_layer
        torch.manual_seed(0)
        T = 8
        q = torch.randn(T, layer.num_heads, 128, dtype=torch.bfloat16)
        flat = torch.arange(T)
        tok = torch.arange(T) + 3
        page = torch.tensor([0, 0, 1, 1, -1, -1, 2, 2])

        cos, sin = layer._blended_cos_sin(flat, tok, page, q)
        q4 = q.transpose(0, 1).unsqueeze(0)
        from vllm.model_executor.models.unlimited_ocr2.dash_router import (
            apply_rotary_pos_emb,
        )

        blended, _ = apply_rotary_pos_emb(
            q4, q4.clone(), cos.unsqueeze(0), sin.unsqueeze(0)
        )
        blended = blended.squeeze(0).transpose(0, 1).float()

        # page2d tokens: rotate-half with the 2-axis cos/sin.
        cos2, sin2 = layer.rope2d(q, tok.unsqueeze(0), page.clamp(min=0).unsqueeze(0))
        half = 64

        def rotate_half(x):
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        for t in (0, 2, 6):
            expected = (
                q[t].float() * cos2[0, t].float()
                + rotate_half(q[t].float()) * sin2[0, t].float()
            )
            torch.testing.assert_close(blended[t], expected, atol=1e-2, rtol=1e-2)

        # 1D fallback tokens (page == -1): must equal the stock vLLM rope op.
        pos_1d = flat[page < 0]
        q_1d = q[page < 0].reshape(-1, layer.num_heads * 128).clone()
        k_1d = torch.zeros_like(q_1d)
        q_ref, _ = layer.rotary_emb(pos_1d, q_1d, k_1d)
        torch.testing.assert_close(
            blended[page < 0].reshape(-1, layer.num_heads * 128),
            q_ref.float(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_cache_writes_land_in_paged_layout(self, dash_layer):
        layer = dash_layer
        num_blocks, H, D = 8, layer.num_kv_heads, 128
        kv = torch.zeros(num_blocks, H, BLOCK, 2 * D, dtype=torch.bfloat16)
        idx_kv = torch.zeros(num_blocks, 1, BLOCK, 128, dtype=torch.float32)
        layer.attn.bind_kv_cache(kv)
        layer.idx_cache.bind_kv_cache(idx_kv)

        block_table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
        seq_lens, context_lens = [70, 40], [0, 10]
        query_lens = [s - c for s, c in zip(seq_lens, context_lens)]
        md = _make_metadata(
            query_lens,
            seq_lens,
            block_table,
            _slot_mapping(block_table, seq_lens, context_lens),
        )
        md_idx = _make_metadata(
            query_lens,
            seq_lens,
            block_table,
            _slot_mapping(block_table, seq_lens, context_lens),
        )

        torch.manual_seed(1)
        T = sum(query_lens)
        k = torch.randn(T, H, D, dtype=torch.bfloat16)
        v = torch.randn(T, H, D, dtype=torch.bfloat16)
        idx_k = torch.randn(T, 128)
        layer._write_caches(k, v, idx_k, md, md_idx)

        # Every new token's K/V landed at its (block, head, offset) cell.
        row = 0
        for r, (seq_len, ctx) in enumerate(zip(seq_lens, context_lens)):
            for pos in range(ctx, seq_len):
                blk = int(block_table[r, pos // BLOCK])
                off = pos % BLOCK
                assert torch.equal(kv[blk, :, off, :D], k[row])
                assert torch.equal(kv[blk, :, off, D:], v[row])
                assert torch.equal(idx_kv[blk, 0, off], idx_k[row])
                row += 1

        # The kernel views expose the same content as [pages, 32, H, D].
        k_cache, v_cache = layer._kernel_kv_views()
        assert k_cache.shape == (num_blocks, BLOCK, H, D)
        assert torch.equal(k_cache[0, 5, 3], kv[0, 3, 5, :D])
        assert torch.equal(v_cache[6, 7, 2], kv[6, 2, 7, D:])

    def test_route_matches_direct_router_calls(self, dash_layer):
        layer = dash_layer
        router = layer.dash_router
        num_blocks = 16
        idx_kv = torch.randn(num_blocks, 1, BLOCK, 128, dtype=torch.float32)
        layer.attn.bind_kv_cache(
            torch.zeros(
                num_blocks, layer.num_kv_heads, BLOCK, 256, dtype=torch.bfloat16
            )
        )
        layer.idx_cache.bind_kv_cache(idx_kv)

        block_table = torch.tensor(
            [[3, 1, 4, 0, 0], [5, 2, 0, 0, 0]], dtype=torch.int32
        )
        seq_lens, query_lens = [100, 40], [100, 40]
        md = _make_metadata(
            query_lens,
            seq_lens,
            block_table,
            _slot_mapping(block_table, seq_lens, [0, 0]),
        )
        md_idx = _make_metadata(
            query_lens,
            seq_lens,
            block_table,
            _slot_mapping(block_table, seq_lens, [0, 0]),
        )

        torch.manual_seed(2)
        T = sum(query_lens)
        idx_q = torch.randn(T, 128)
        flat = torch.cat([torch.arange(s) for s in seq_lens])
        # req0 queries sit on page 0, req1 queries have no page.
        page_pos = torch.cat(
            [torch.zeros(seq_lens[0], dtype=torch.long), torch.full((seq_lens[1],), -1)]
        )
        # req0's image tokens live on block 1 (token range 32..64, page 0).
        img_page_tok = torch.full((2, 128), -1, dtype=torch.int32)
        img_page_tok[0, 33:60] = 0
        from vllm.model_executor.models.unlimited_ocr2.attention import (
            DashContextProvider,
        )

        provider = DashContextProvider()
        provider.dash_ctx = DashContext(img_page_tok)
        layer.dash_ctx_provider = provider

        topk_idx, blk_bias = layer._route(idx_q, flat, page_pos, md, md_idx)
        topk = router.route_params.topk
        assert topk_idx.shape == (1, T, topk)
        assert topk_idx.dtype == torch.int32
        assert blk_bias.shape == (1, T, topk)

        qsl = [0, *torch.tensor(query_lens).cumsum(0).tolist()]
        for r, seq_len in enumerate(seq_lens):
            nblk = -(-seq_len // BLOCK)
            pages = block_table[r, :nblk].long()
            idx_k_full = idx_kv[pages, 0].reshape(nblk * BLOCK, -1)[:seq_len]
            k_bar, chunk_valid = router.head_cls_summary(idx_k_full.unsqueeze(0), nblk)
            scores = router.block_scores(
                idx_q[qsl[r] : qsl[r + 1]].unsqueeze(0),
                flat[qsl[r] : qsl[r + 1]].unsqueeze(0),
                nblk,
                k_bar=k_bar,
                chunk_valid=chunk_valid,
            )
            img_blk = torch.nn.functional.pad(
                img_page_tok[r, :seq_len], (0, nblk * BLOCK - seq_len), value=-1
            ).view(nblk, BLOCK)
            qp = (page_pos[qsl[r] : qsl[r + 1]] - 1)[:, None, None]
            cand = ((qp >= 0) & (qp == img_blk.unsqueeze(0))).any(-1).unsqueeze(0)
            topi, bias, valid = router.route_and_select(scores, cand)
            k_sel = topi.shape[-1]
            expected = torch.where(valid[0], topi[0].to(torch.int32), -1)
            assert torch.equal(topk_idx[0, qsl[r] : qsl[r + 1], :k_sel], expected)
            if k_sel < topk:
                assert (topk_idx[0, qsl[r] : qsl[r + 1], k_sel:] == -1).all()
            assert torch.equal(blk_bias[0, qsl[r] : qsl[r + 1], :k_sel], bias[0])

        # Page-0 queries of req0 can only read the block holding page-0
        # image tokens (block index 1) once they are past the local window.
        sel = topk_idx[0, -1][topk_idx[0, -1] >= 0]
        assert set(sel.tolist()) <= {1}

    def test_route_without_context_selects_remote_blocks(self, dash_layer):
        layer = dash_layer
        num_blocks = 8
        layer.attn.bind_kv_cache(
            torch.zeros(
                num_blocks, layer.num_kv_heads, BLOCK, 256, dtype=torch.bfloat16
            )
        )
        layer.idx_cache.bind_kv_cache(
            torch.randn(num_blocks, 1, BLOCK, 128, dtype=torch.float32)
        )
        from vllm.model_executor.models.unlimited_ocr2.attention import (
            DashContextProvider,
        )

        layer.dash_ctx_provider = DashContextProvider()

        seq_lens, query_lens = [200], [4]
        context = [196]
        block_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
        md = _make_metadata(
            query_lens,
            seq_lens,
            block_table,
            _slot_mapping(block_table, seq_lens, context),
        )
        md_idx = _make_metadata(
            query_lens,
            seq_lens,
            block_table,
            _slot_mapping(block_table, seq_lens, context),
        )
        torch.manual_seed(3)
        idx_q = torch.randn(sum(query_lens), 128)
        flat = torch.arange(context[0], seq_lens[0])
        page_pos = torch.zeros(sum(query_lens), dtype=torch.long)
        topk_idx, _ = layer._route(idx_q, flat, page_pos, md, md_idx)

        # Queries at positions 196..199: only blocks fully below the local
        # window (block <= q_block - 4) are eligible; -1 pads the rest.
        for i, pos in enumerate(range(context[0], seq_lens[0])):
            sel = topk_idx[0, i][topk_idx[0, i] >= 0]
            max_remote_block = pos // BLOCK - layer.local_blocks
            assert sel.numel() <= max(0, max_remote_block + 1)
            assert set(sel.tolist()) <= set(range(max_remote_block + 1))


class _DashRouterSanity:
    """The real DashRouter is constructed by the layer with the checkpoint's
    routing parameters."""

    def test_router_params_from_config(self, dash_layer):
        router: DashRouter = dash_layer.dash_router
        p = router.route_params
        assert p.topk == 12
        assert p.topm == 16
        assert p.idx_scale == pytest.approx(128.0 / math.sqrt(128))
        assert p.floor == pytest.approx(1e-4)
        assert p.sigma == pytest.approx(1.0)
