# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import math
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models.unlimited_ocr2 import (
    DashRouter,
    Page2DRotaryEmbedding,
    PageRopeState,
    build_page_aware_ids,
    build_page_candidate_mask,
    build_page_rope_positions,
    entmax15,
)

REF_DIR = Path("/tmp/uocr2")
pytestmark = pytest.mark.skipif(
    not (REF_DIR / "dash_attention.py").exists(),
    reason="Unlimited-OCR-2 HF reference copy not present at /tmp/uocr2",
)

IMG_ID = 128815
PAGE_ID = 128827


def _load_ref_defs(filename, names, extra_ns):
    """Exec reference defs verbatim, bypassing the file's heavy imports.

    _sparse_base_impl.py / modeling_deepseekv2.py cannot be imported in this
    environment (they pin an older transformers), so the exact reference
    functions are lifted out of the source with ast and executed against a
    namespace holding only their true dependencies.
    """
    path = REF_DIR / filename
    tree = ast.parse(path.read_text())
    wanted = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            wanted.setdefault(node.name, node)
    missing = set(names) - set(wanted)
    assert not missing, f"{missing} not found in {path}"
    module = ast.Module(body=[wanted[n] for n in names], type_ignores=[])
    ns = dict(extra_ns)
    exec(compile(module, str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def ref_dash():
    sys.path.insert(0, str(REF_DIR))
    try:
        import dash_attention
    finally:
        sys.path.remove(str(REF_DIR))
    return dash_attention


def _make_ref_router(ref_dash, stage0):
    """Real reference DashAttention instance with a minimal HF config."""
    old = os.environ.get("DASH_STAGE0")
    os.environ["DASH_STAGE0"] = stage0
    try:
        cfg = types.SimpleNamespace(
            sparse_attention_config={
                "sparse_block_size": 32,
                "sparse_local_block": 4,
                "sparse_index_dim": 128,
                "dash_sigma": 1.0,
                "dash_query_chunk": 64,
            },
            num_attention_heads=10,
            hidden_size=1280,
            rms_norm_eps=1e-6,
        )
        router = ref_dash.DashAttention(cfg)
    finally:
        if old is None:
            del os.environ["DASH_STAGE0"]
        else:
            os.environ["DASH_STAGE0"] = old
    return router


def _sync_weights(ref_router, router):
    with torch.no_grad():
        router.q_proj.weight.copy_(ref_router.q_proj.weight)
        router.k_proj.weight.copy_(ref_router.k_proj.weight)
        router.q_norm.weight.copy_(ref_router.q_norm.weight)
        router.k_norm.weight.copy_(ref_router.k_norm.weight)
        router.head_cls.copy_(torch.randn_like(router.head_cls))
        ref_router.head_cls.copy_(router.head_cls)


def _prompt_ids(img_counts=(40, 55), pre=7, mid=5, instr=6, resp=(9, 4)):
    """Synthetic two-page document: text, <PAGE> img text, <PAGE> img,
    instruction, response text, <PAGE>, response text."""
    ids = [1] + list(range(100, 100 + pre))
    for k, n_img in enumerate(img_counts):
        ids.append(PAGE_ID)
        ids += [IMG_ID] * n_img
        if k == 0:
            ids += list(range(300, 300 + mid))
    ids += list(range(400, 400 + instr))
    ids += list(range(500, 500 + resp[0]))
    ids.append(PAGE_ID)
    ids += list(range(600, 600 + resp[1]))
    return torch.tensor([ids], dtype=torch.long)


def test_entmax15_parity(ref_dash):
    torch.manual_seed(0)
    for shape in [(2, 7, 33), (3, 1), (1, 5, 16, 12)]:
        x = torch.randn(*shape) * 3.0
        x[..., -2:] = -1.0e4
        ref = ref_dash.entmax15(x, dim=-1)
        got = entmax15(x, dim=-1)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_route_and_select_parity(ref_dash):
    ref_ns = _load_ref_defs(
        "modeling_deepseekv2.py",
        ["_infer_route_select"],
        {
            "torch": torch,
            "_SCORE_NEG": ref_dash._SCORE_NEG,
            "_ENTMAX_NEG": -1.0e4,
            "_entmax15": ref_dash.entmax15,
        },
    )
    ref_select = ref_ns["_infer_route_select"]

    torch.manual_seed(1)
    B, Lq, nblk = 2, 37, 29
    scores = torch.randn(B, Lq, nblk)
    remote = torch.rand(B, Lq, nblk) > 0.3
    scores = scores.masked_fill(~remote, ref_dash._SCORE_NEG)
    cand = torch.rand(B, Lq, nblk) > 0.4
    cand[0, 3] = False
    cand[1, 10] = True
    p = dict(idx_scale=128.0 / math.sqrt(128), floor=1e-4, sigma=1.0, K=12, topm=16)

    router = DashRouter(
        hidden_size=1280, index_dim=128, block_size=32, local_blocks=4, q_chunk=64
    )
    router.set_inference_params(
        idx_scale=p["idx_scale"],
        floor=p["floor"],
        topk=p["K"],
        topm=p["topm"],
        sigma=p["sigma"],
    )

    for bc in (cand, None):
        ri, rb, rv = ref_select(scores, bc, p)
        gi, gb, gv = router.route_and_select(scores, bc)
        torch.testing.assert_close(gi, ri, rtol=0, atol=0)
        torch.testing.assert_close(gb, rb, rtol=0, atol=0)
        torch.testing.assert_close(gv, rv, rtol=0, atol=0)


@pytest.mark.parametrize("stage0", ["head_cls", "indexer"])
def test_router_scoring_parity(ref_dash, stage0):
    ref_ns = _load_ref_defs(
        "modeling_deepseekv2.py",
        ["_router_qk", "_block_scores_infer"],
        {
            "torch": torch,
            "F": F,
            "math": math,
            "_llama_apply_rotary_pos_emb": None,
            "_SCORE_NEG": ref_dash._SCORE_NEG,
            "_ACT_CLAMP": ref_dash._ACT_CLAMP,
        },
    )

    torch.manual_seed(2)
    ref_router = _make_ref_router(ref_dash, stage0)
    router = DashRouter(
        hidden_size=1280,
        index_dim=128,
        block_size=32,
        local_blocks=4,
        q_chunk=64,
        stage0=stage0,
    )
    _sync_weights(ref_router, router)

    B, S, hidden = 2, 150, 1280
    hidden_states = torch.randn(B, S, hidden)
    pos = torch.arange(S)[None].expand(B, -1)
    nblk = (S + 31) // 32

    ref_q, ref_k = ref_ns["_router_qk"](ref_router, hidden_states, None)
    got_q, got_k = router.index_qk(hidden_states)
    torch.testing.assert_close(got_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=0)

    ref_scores = ref_ns["_block_scores_infer"](ref_router, ref_q, ref_k, pos, nblk)
    if stage0 == "head_cls":
        k_bar, chunk_valid = router.head_cls_summary(got_k, nblk)
        got_scores = router.block_scores(
            got_q, pos, nblk, k_bar=k_bar, chunk_valid=chunk_valid
        )
    else:
        got_scores = router.block_scores(got_q, pos, nblk, idx_k=got_k)
    torch.testing.assert_close(got_scores, ref_scores, rtol=0, atol=0)


def test_page2d_rotary_parity():
    ref_ns = _load_ref_defs(
        "_sparse_base_impl.py",
        ["DeepseekV2IndexerRotary2D"],
        {"torch": torch, "nn": nn, "math": math},
    )
    ref_cls = ref_ns["DeepseekV2IndexerRotary2D"]

    for mode in ("resample", "truncate"):
        ref = ref_cls(128, 1e4, 256.0, 1, tok_freq_mode=mode, page_mod=2)
        got = Page2DRotaryEmbedding(128, 1e4, 256.0, 1, tok_freq_mode=mode, page_mod=2)
        torch.testing.assert_close(got.inv_tok, ref.inv_tok, rtol=0, atol=0)
        torch.testing.assert_close(got.inv_page, ref.inv_page, rtol=0, atol=0)

        x = torch.zeros(2, 11, 128, dtype=torch.bfloat16)
        tok_pos = torch.arange(11)[None].expand(2, -1)
        page_pos = torch.tensor([[0, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2]] * 2)
        ref_cos, ref_sin = ref(x, tok_pos, page_pos)
        got_cos, got_sin = got(x, tok_pos, page_pos)
        torch.testing.assert_close(got_cos, ref_cos, rtol=0, atol=0)
        torch.testing.assert_close(got_sin, ref_sin, rtol=0, atol=0)


def _ref_page_self():
    """Attribute carrier for the extracted reference methods (which take
    ``self``); the HF indexer class itself is not importable here."""
    ref_ns = _load_ref_defs(
        "_sparse_base_impl.py",
        [
            "_build_page_rope_pos",
            "_page_rope_offsets",
            "_build_page_aware_ids",
            "_build_kl_candidate_mask",
            "_build_page_aware_image_block_mask",
            "_build_image_block_allow_mask",
            "_main_rope_coords",
        ],
        {
            "torch": torch,
            "F": F,
            "math": math,
            "_PAGE_SCOPE_PREFILL": True,
        },
    )
    self = types.SimpleNamespace(
        image_token_id=IMG_ID,
        page_token_id=PAGE_ID,
        block_size=32,
        rope_page_span=0,
        per_page_image_tokens=273,
        image_bucket_constraint=True,
        layer_idx=0,
        q_proj=nn.Linear(1, 1),
        rotary_emb_2d=types.SimpleNamespace(base_page=256.0, page_pairs=1, page_mod=2),
        _page_rope_bail=lambda *a: None,
    )
    for name in (
        "_build_page_rope_pos",
        "_page_rope_offsets",
        "_build_page_aware_ids",
        "_build_kl_candidate_mask",
        "_build_page_aware_image_block_mask",
        "_build_image_block_allow_mask",
        "_main_rope_coords",
    ):
        setattr(self, name, types.MethodType(ref_ns[name], self))
    return self


def test_page_rope_positions_parity():
    self = _ref_page_self()
    ids = _prompt_ids()
    S = ids.shape[-1]

    ref_tok, ref_page = self._build_page_rope_pos(ids, S)
    out = build_page_rope_positions(ids, IMG_ID, PAGE_ID)
    assert out is not None
    got_tok, got_page, got_pf_cnt, got_n_pg = out
    torch.testing.assert_close(got_tok, ref_tok, rtol=0, atol=0)
    torch.testing.assert_close(got_page, ref_page, rtol=0, atol=0)
    torch.testing.assert_close(
        got_pf_cnt, self._page_rope_meta["pf_cnt"], rtol=0, atol=0
    )
    assert got_n_pg == self._page_rope_meta["n_pg"]

    # Reference bails to None without image/page markers; so does the port.
    text_only = torch.tensor([[1, 5, 6, 7]])
    assert self._build_page_rope_pos(text_only, 4) is None
    assert build_page_rope_positions(text_only, IMG_ID, PAGE_ID) is None


def test_page_aware_ids_parity():
    self = _ref_page_self()
    ids = _prompt_ids()
    S = ids.shape[-1]

    ref = self._build_page_aware_ids(ids, IMG_ID, PAGE_ID, S)
    got = build_page_aware_ids(ids, IMG_ID, PAGE_ID)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, rtol=0, atol=0)


def test_page_candidate_mask_parity():
    self = _ref_page_self()
    ids = _prompt_ids()
    S = ids.shape[-1]
    itm, ipi, qpi = self._build_page_aware_ids(ids, IMG_ID, PAGE_ID, S)
    nblk = (S + 31) // 32
    pos = torch.arange(S)[None]
    B = 1

    cases = [
        dict(
            image_token_mask=itm,
            image_token_page_ids=ipi,
            query_token_page_ids=qpi,
            page_isolation=True,
        ),
        dict(image_token_mask=itm, page_isolation=False),
        dict(),
    ]
    for kw in cases:
        ref = self._build_kl_candidate_mask(
            pos,
            nblk,
            kw.get("image_token_mask"),
            B,
            S,
            image_token_page_ids=kw.get("image_token_page_ids"),
            query_token_page_ids=kw.get("query_token_page_ids"),
        )[:, 0]
        got = build_page_candidate_mask(pos, nblk, 32, **kw)
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_page_state_advance_parity():
    self = _ref_page_self()
    ids = _prompt_ids()
    S = ids.shape[-1]

    ref_tok, ref_page = self._build_page_rope_pos(ids, S)
    pf_cnt = self._page_rope_meta["pf_cnt"]
    n_pg = self._page_rope_meta["n_pg"]
    state = PageRopeState.from_prefill(ref_tok, ref_page, pf_cnt, n_pg)

    ref_st = {
        "page": ref_page[:, -1].clone(),
        "tok": ref_tok[:, -1].clone(),
        "pf_cnt": pf_cnt,
        "n_pg": n_pg,
    }
    past_kv = types.SimpleNamespace(_page_rope_state={0: ref_st})
    self.indexer = types.SimpleNamespace(page_token_id=PAGE_ID)
    self._main_rope_pos_2d = None

    decode_ids = [700, 701, PAGE_ID, 702, 703, PAGE_ID, 704]
    running = ids
    for tok in decode_ids:
        running = torch.cat([running, torch.tensor([[tok]], dtype=torch.long)], dim=1)
        self._sparse_source_input_ids = running
        ref_coords = self._main_rope_coords(1, past_kv)
        got_tok, got_page = state.advance(running, PAGE_ID)
        torch.testing.assert_close(got_tok, ref_coords[0], rtol=0, atol=0)
        torch.testing.assert_close(got_page, ref_coords[1], rtol=0, atol=0)
