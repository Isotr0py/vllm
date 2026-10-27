# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""page2d dual-axis RoPE for baidu/Unlimited-OCR-2.

Framework-agnostic port of the Unlimited-OCR-2 HF reference
(``_sparse_base_impl.py``): the two-axis (intra-page offset, page id) rotary
embedding ``DeepseekV2IndexerRotary2D``, the page-coordinate construction
``_build_page_rope_pos`` / ``_page_rope_offsets``, the page-aware id builders
``_build_page_aware_ids``, and the page-aware routing candidate mask
``_build_kl_candidate_mask``.

Layout: ``head_dim = 128`` gives 64 frequency pairs; ``page_pairs`` of them
(here 1, theta=256) encode the page axis (folded by ``page_mod``, here 2, so
only page parity is expressed) and the remaining 63 token pairs are resampled
over the full theta=1e4 spectrum. Page 0 is a reserved bucket for the
BOS/system prefix and the trailing instruction; real pages start at 1, so the
instruction can never look "same page" as any page's image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Page2DRotaryEmbedding(nn.Module):
    """2-axis RoPE over (tok_pos, page_pos), frequency-pair space.

    Port of ``DeepseekV2IndexerRotary2D``. Pairs ``[0, n_tok)`` drive the
    token axis at ``base_tok``, pairs ``[n_tok, half)`` the page axis at
    ``base_page``; ``page_mod > 0`` folds the page coordinate to parity
    (``page_pos % page_mod``) at the angle computation only.
    """

    def __init__(
        self,
        dim: int,
        base_tok: float,
        base_page: float,
        page_pairs: int,
        tok_freq_mode: str = "resample",
        page_mod: int = 0,
    ):
        super().__init__()
        half = dim // 2
        page_pairs = int(page_pairs)
        if not 0 < page_pairs < half:
            raise ValueError(
                f"page_pairs={page_pairs} must be in (0, {half}); dim={dim}"
            )
        n_tok = half - page_pairs
        ar = torch.arange(0, n_tok, dtype=torch.int64).float()
        if str(tok_freq_mode).lower() == "truncate":
            inv_tok = 1.0 / (base_tok ** (ar * 2 / dim))
        else:
            inv_tok = 1.0 / (base_tok ** (ar / n_tok))
        inv_page = 1.0 / (
            base_page
            ** (torch.arange(0, page_pairs, dtype=torch.int64).float() / page_pairs)
        )
        self.register_buffer("inv_tok", inv_tok, persistent=False)
        self.register_buffer("inv_page", inv_page, persistent=False)
        self.dim = dim
        self.n_tok = n_tok
        self.page_pairs = page_pairs
        self.base_tok = base_tok
        self.base_page = base_page
        self.tok_freq_mode = str(tok_freq_mode).lower()
        self.page_mod = int(page_mod or 0)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        tok_pos: torch.Tensor,
        page_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the 2-axis cos/sin.

        Args:
            x: reference tensor for device/dtype.
            tok_pos: [B, S] intra-page running offsets.
            page_pos: [B, S] page ids (page 0 = reserved no-page bucket).

        Returns:
            (cos, sin), each [B, S, dim] in ``x.dtype``, same layout as 1D.

        """
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            t = tok_pos.to(device=x.device).float()[:, :, None]
            p = page_pos.to(device=x.device)
            if self.page_mod > 0:
                p = torch.remainder(p.to(torch.int64), self.page_mod)
            p = p.float()[:, :, None]
            a_tok = t * self.inv_tok.to(x.device).float()[None, None, :]
            a_page = p * self.inv_page.to(x.device).float()[None, None, :]
            ang = torch.cat((a_tok, a_page), dim=-1)
            emb = torch.cat((ang, ang), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _page_rope_offsets(
    page: torch.Tensor,
    in_prefill: torch.Tensor,
    rope_page_span: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Running-count intra-page offsets plus per-page prefill lengths.

    A running count (not index - segment_start) is required because one page
    owns several disjoint regions (its prompt image run plus its response text
    run), so the response text of page k starts at that page's prefill length
    and variable per-page image counts need no fixed-page-size assumption.

    Returns:
        (tok_pos [B, S] long, pf_cnt [B, n_pg] long, n_pg).

    """
    B, S = page.shape
    n_pg = int(page.max().item()) + 1
    onehot = F.one_hot(page, num_classes=n_pg).to(torch.int32)
    run = onehot.cumsum(dim=1)
    tok = run.gather(2, page[:, :, None]).squeeze(-1).to(torch.int64) - 1

    pf_cnt = (onehot * in_prefill[:, :, None].to(torch.int32)).sum(dim=1)
    pf_cnt = pf_cnt.to(torch.int64)
    if rope_page_span > 0:
        pf_k = pf_cnt.gather(1, page.reshape(B, S))
        s_span = float(rope_page_span)
        pre = torch.round(tok.float() * s_span / pf_k.clamp(min=1).float())
        post = s_span + (tok - pf_k).float()
        scaled = torch.where(in_prefill, pre, post).to(torch.int64)
        tok = torch.where(page > 0, scaled, tok)
    return tok, pf_cnt, n_pg


def build_page_rope_positions(
    input_ids: torch.Tensor | None,
    image_token_id: int | None,
    page_token_id: int | None,
    rope_page_span: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Build (tok_pos, page_pos) for the 2-axis RoPE from prompt token ids.

    page_pos: prefill rows take the raw ``<PAGE>`` marker count (1..N);
    response rows take ``count - N`` so image page k and text page k share the
    id. Everything with no page (BOS/system prefix, trailing instruction)
    gets page 0, the reserved bucket. tok_pos is the running count inside the
    token's own page bucket.

    Args:
        input_ids: [B, S] prompt token ids.
        image_token_id: id of the image placeholder token.
        page_token_id: id of the ``<PAGE>`` marker token.
        rope_page_span: > 0 maps each page's prefill span onto a fixed length
            (ablation in the reference; Unlimited-OCR-2 ships 0).

    Returns:
        (tok_pos [B, S] long, page_pos [B, S] long, pf_cnt [B, n_pg] long,
        n_pg), or None when the row has no image/page markers (caller falls
        back to 1D RoPE).

    """
    if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
        return None
    if image_token_id is None or page_token_id is None:
        return None
    ids = input_ids
    B, S = ids.shape
    ar = torch.arange(S, device=ids.device)
    img = ids == image_token_id
    mk = ids == page_token_id
    if not bool(img.any()) or not bool(mk.any()):
        return None

    cum = torch.cumsum(mk.long(), dim=-1)
    last_img = (img.long() * (ar + 1)).max(dim=-1).values - 1
    n_pages = cum.gather(1, last_img.clamp(min=0)[:, None])
    n_pages = n_pages * (last_img >= 0)[:, None].long()
    if not bool((n_pages > 0).all()):
        return None

    in_prefill = ar[None, :] <= last_img[:, None]
    page = torch.where(in_prefill, cum, cum - n_pages).clamp(min=0)
    tok, pf_cnt, n_pg = _page_rope_offsets(page, in_prefill, rope_page_span)
    return tok, page, pf_cnt, n_pg


def build_page_aware_ids(
    input_ids: torch.Tensor | None,
    image_token_id: int | None,
    page_token_id: int | None,
    per_page_image_tokens: int = 273,
    page_scope_prefill: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Build per-token image masks and page ids for page-aware routing.

    Port of ``_build_page_aware_ids`` (marker mode): image tokens are bucketed
    by the ``<PAGE>`` markers the prompt carries (falling back to
    ``per_page_image_tokens`` chunks when the prompt has none); query page ids
    continue the count into the response, with page-scope prefill (prompt
    rows of page k carry page id k; the instruction after the last image
    stays -1, a global query).

    Returns:
        (image_token_mask [B, S] bool, image_token_page_ids [B, S] long with
        -1 off-image, query_token_page_ids [B, S] long with -1 = no page), or
        (None, None, None) when page-aware routing is unavailable.

    """
    if input_ids is None or image_token_id is None or page_token_id is None:
        return None, None, None
    if input_ids.ndim != 2:
        return None, None, None

    token_ids = input_ids
    image_token_mask = token_ids == image_token_id
    page_starts = token_ids == page_token_id
    cum_pages = torch.cumsum(page_starts.long(), dim=-1)

    seq = token_ids.shape[-1]
    ar = torch.arange(seq, device=token_ids.device)
    last_img = (image_token_mask.long() * (ar + 1)).max(dim=-1).values - 1
    prompt_pages = cum_pages.gather(1, last_img.clamp(min=0)[:, None])
    prompt_pages = prompt_pages * (last_img >= 0)[:, None].long()
    marker_mode = prompt_pages > 0

    per_page = max(1, int(per_page_image_tokens))
    img_running = torch.cumsum(image_token_mask.long(), dim=-1) - 1
    image_token_page_ids = torch.where(
        marker_mode, cum_pages - 1, img_running // per_page
    )
    image_token_page_ids = image_token_page_ids.masked_fill(~image_token_mask, -1)

    raw_page = cum_pages - 1
    resp_page = raw_page - torch.where(
        marker_mode, prompt_pages, torch.zeros_like(prompt_pages)
    )
    in_prefill = ar[None, :] <= last_img[:, None]
    query_token_page_ids = torch.where(in_prefill, raw_page, resp_page)
    if not page_scope_prefill:
        query_token_page_ids = query_token_page_ids.masked_fill(in_prefill, -1)
    query_token_page_ids = query_token_page_ids.masked_fill(
        query_token_page_ids < 0, -1
    )

    if image_token_mask.sum() == 0 or (query_token_page_ids >= 0).sum() == 0:
        return None, None, None
    return image_token_mask, image_token_page_ids, query_token_page_ids


def build_image_block_mask(
    image_token_mask: torch.Tensor,
    nblk: int,
    block_size: int,
) -> torch.Tensor:
    """[B, nblk] bool: the block contains >= 1 image token."""
    B, k_len = image_token_mask.shape
    pad = nblk * block_size - k_len
    if pad > 0:
        mask = F.pad(image_token_mask.to(torch.bool), (0, pad), value=False)
    elif pad < 0:
        mask = image_token_mask.to(torch.bool)[..., : nblk * block_size]
    else:
        mask = image_token_mask.to(torch.bool)
    return mask.view(B, nblk, block_size).any(dim=-1)


def build_page_candidate_mask(
    query_positions: torch.Tensor,
    nblk: int,
    block_size: int,
    image_token_mask: torch.Tensor | None = None,
    image_token_page_ids: torch.Tensor | None = None,
    query_token_page_ids: torch.Tensor | None = None,
    page_isolation: bool = True,
) -> torch.Tensor:
    """Per-query remote-block candidate domain [B, Lq, nblk] bool.

    Port of ``_build_kl_candidate_mask``: with page isolation and page ids,
    candidates are the same-page image blocks (queries with no page, id -1,
    see none); with only the flat image mask, all image blocks; with no image
    info, all blocks. A query that sees no candidate at all falls back to all
    blocks.
    """
    B, Lq = query_positions.shape
    device = query_positions.device
    if (
        page_isolation
        and image_token_page_ids is not None
        and query_token_page_ids is not None
    ):
        k_len = image_token_page_ids.shape[-1]
        pos_clamped = query_positions.clamp(0, query_token_page_ids.shape[-1] - 1)
        query_pages = query_token_page_ids.gather(1, pos_clamped)
        pad = nblk * block_size - k_len
        if pad > 0:
            img_pages = F.pad(image_token_page_ids.float(), (0, pad), value=-1).long()
        elif pad < 0:
            img_pages = image_token_page_ids[..., : nblk * block_size]
        else:
            img_pages = image_token_page_ids
        img_page_blocks = img_pages.view(B, nblk, block_size)
        qp = query_pages[:, :, None, None]
        same_page = (qp >= 0) & (qp == img_page_blocks[:, None])
        candidate = same_page.any(dim=-1)
    elif image_token_mask is not None:
        img_blk = build_image_block_mask(image_token_mask, nblk, block_size)
        candidate = img_blk[:, None, :].expand(B, Lq, nblk).contiguous()
    else:
        candidate = torch.ones(B, Lq, nblk, dtype=torch.bool, device=device)
    has_candidate = candidate.any(dim=-1, keepdim=True)
    return torch.where(has_candidate, candidate, torch.ones_like(candidate))
