# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-side page2d RoPE state for baidu/Unlimited-OCR-2.

The DASH language model consumes four position channels per token (see
``PAGE_ROPE_*`` in ``vllm/model_executor/models/unlimited_ocr2/attention.py``):
flat 1D position, intra-page token offset, page id (-1 = page2d inactive),
and the image page id used for page-aware routing.

Prefill channels are built once per request from the full prompt token ids
(:func:`build_page_rope_positions` / :func:`build_page_aware_ids`, the same
parity-tested helpers the model uses) and staged into a persistent buffer, so
chunked prefill reads slices and the page-aware routing context can gather
the image page ids over the whole KV range.

Decode coordinates come from an incremental state machine (one row per
request slot): each decode step consumes the step's token ids -- a ``<PAGE>``
marker advances the request to the next page and resets the token offset to
that page's prefill length, any other token increments the offset. This
mirrors the HF reference ``_main_rope_coords`` decode branch and the
model-side ``PageRopeState``.

The class duck-types the ``RopeState`` interface used by ``DefaultModelState``
(``init_prefill_positions`` / ``apply_staged_writes`` / ``prepare_positions``
/ ``get_positions``) but keeps plain torch buffers instead of the UVA/triton
staging machinery, so the whole flow also runs on CPU.
"""

import torch
import torch.nn as nn

from vllm.model_executor.models.unlimited_ocr2.attention import (
    PAGE_ROPE_FLAT_CH,
    PAGE_ROPE_IMG_CH,
    PAGE_ROPE_NUM_DIMS,
    PAGE_ROPE_PAGE_CH,
    PAGE_ROPE_TOK_CH,
    DashContext,
)
from vllm.model_executor.models.unlimited_ocr2.page2d_rope import (
    build_page_aware_ids,
    build_page_rope_positions,
)
from vllm.v1.worker.gpu.mm.rope import RopeState

_DEFAULT_MAX_PAGES = 1024


class Page2dRopeState(RopeState):
    """Per-request page2d position channels and decode state machine."""

    def __init__(
        self,
        *,
        image_token_id: int,
        page_token_id: int,
        per_page_image_tokens: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ):
        # Deliberately skips RopeState.__init__: that builds the UVA/triton
        # staging machinery, which is unavailable on CPU and unnecessary
        # here -- page prompts are staged once per request, so plain torch
        # buffers and writes are fast enough and keep the flow CPU-testable.
        self.num_dims = PAGE_ROPE_NUM_DIMS
        self.image_token_id = image_token_id
        self.page_token_id = page_token_id
        self.per_page_image_tokens = per_page_image_tokens
        self.max_pages = max_pages
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.device = device

        # Staged prefill channels: one row per (request slot, channel).
        self.channels = torch.full(
            (max_num_reqs * PAGE_ROPE_NUM_DIMS, max_model_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.positions = torch.zeros(
            (PAGE_ROPE_NUM_DIMS, max_num_tokens + 1),
            dtype=torch.int64,
            device=device,
        )

        # Decode state: device-resident source of truth.
        self.dec_page = torch.full(
            (max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.dec_tok = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.dec_npg = torch.ones(max_num_reqs, dtype=torch.int32, device=device)
        self.dec_pfcnt = torch.zeros(
            (max_num_reqs, max_pages), dtype=torch.int32, device=device
        )
        # Host staging, flushed in apply_staged_writes().
        self._staged_channels: list[tuple[int, torch.Tensor]] = []
        self._staged_state: list[
            tuple[int, int, int, int, torch.Tensor]
        ] = []  # (req_idx, page, tok, n_pg, pf_cnt)

    def get_positions(self, num_tokens: int) -> torch.Tensor:
        return self.positions[:, :num_tokens]

    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        ids = torch.tensor(prefill_token_ids, dtype=torch.long).unsqueeze(0)
        seq_len = ids.shape[1]
        flat = torch.arange(seq_len, dtype=torch.long)

        res = build_page_rope_positions(ids, self.image_token_id, self.page_token_id)
        if res is None:
            # No page markers (e.g. text-only chat): 1D fallback.
            tok, page = flat, torch.full_like(flat, -1)
            img = torch.full_like(flat, -1)
            seed = (-1, seq_len - 1, 1, torch.zeros(0, dtype=torch.long))
        else:
            tok, page, pf_cnt, n_pg = res
            if n_pg > self.max_pages:
                raise ValueError(
                    f"Prompt has {n_pg - 1} pages, exceeding the page-rope "
                    f"limit of {self.max_pages - 1}."
                )
            _, img_ids, _ = build_page_aware_ids(
                ids,
                self.image_token_id,
                self.page_token_id,
                self.per_page_image_tokens,
            )
            img = img_ids[0] if img_ids is not None else torch.full_like(flat, -1)
            tok, page = tok[0], page[0]
            seed = (int(page[-1]), int(tok[-1]), n_pg, pf_cnt[0])

        channels = torch.stack((flat, tok, page, img.to(torch.long))).to(torch.int32)
        self._staged_channels.append((req_idx, channels))
        self._staged_state.append((req_idx, *seed))

    def apply_staged_writes(self) -> None:
        for req_idx, channels in self._staged_channels:
            base = PAGE_ROPE_NUM_DIMS * req_idx
            self.channels[base : base + PAGE_ROPE_NUM_DIMS].fill_(-1)
            length = channels.shape[1]
            self.channels[base : base + PAGE_ROPE_NUM_DIMS, :length].copy_(
                channels.to(self.device)
            )
        self._staged_channels.clear()
        for req_idx, page, tok, n_pg, pf_cnt in self._staged_state:
            self.dec_page[req_idx] = page
            self.dec_tok[req_idx] = tok
            self.dec_npg[req_idx] = n_pg
            self.dec_pfcnt[req_idx].zero_()
            if pf_cnt.numel():
                self.dec_pfcnt[req_idx, : pf_cnt.numel()] = pf_cnt.to(self.device)
        self._staged_state.clear()

    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        num_tokens: int | None = None,
    ) -> None:
        """Fill the position channels for the current batch.

        Prefill tokens read their staged channels; decode tokens advance the
        per-request state machine (exactly one token per decode row -- the
        model does not support speculative decoding).
        """
        assert input_ids is not None and num_tokens is not None
        device = self.device
        qsl = query_start_loc.long()
        qlens = qsl[1:] - qsl[:-1]
        token_req = torch.repeat_interleave(
            torch.arange(idx_mapping.shape[0], device=device), qlens
        )[:num_tokens]
        total = token_req.shape[0]
        if total == 0:
            return
        rs = idx_mapping.long()[token_req]

        nct = num_computed_tokens[rs].long()
        within = torch.arange(total, device=device) - qsl[token_req]
        flat = nct + within
        is_prefill = nct < prefill_lens[rs].long()
        dec = ~is_prefill

        rows = rs * PAGE_ROPE_NUM_DIMS
        tok_staged = self.channels[rows + PAGE_ROPE_TOK_CH, flat].long()
        page_staged = self.channels[rows + PAGE_ROPE_PAGE_CH, flat].long()
        img_staged = self.channels[rows + PAGE_ROPE_IMG_CH, flat].long()

        # Decode state machine: advance once per decode token.
        page = self.dec_page[rs].long()
        tok = self.dec_tok[rs].long()
        npg = self.dec_npg[rs].long()
        is_pg = (input_ids[:total].long() == self.page_token_id) & dec
        page_new = torch.where(
            is_pg, (page + 1).clamp(max=(npg - 1).clamp(min=0)), page
        )
        pg_start = (
            self.dec_pfcnt[rs]
            .gather(1, page_new.clamp(min=0, max=self.max_pages - 1).unsqueeze(1))
            .squeeze(1)
            .long()
        )
        tok_new = torch.where(is_pg, pg_start, tok + 1)
        dec_rs = rs[dec]
        self.dec_page[dec_rs] = page_new[dec].to(torch.int32)
        self.dec_tok[dec_rs] = tok_new[dec].to(torch.int32)

        pos = self.positions
        pos[PAGE_ROPE_FLAT_CH, :total] = flat
        pos[PAGE_ROPE_TOK_CH, :total] = torch.where(dec, tok_new, tok_staged)
        pos[PAGE_ROPE_PAGE_CH, :total] = torch.where(dec, page_new, page_staged)
        pos[PAGE_ROPE_IMG_CH, :total] = torch.where(
            dec, torch.full_like(img_staged, -1), img_staged
        )

    def build_dash_context(self, idx_mapping: torch.Tensor, s_max: int) -> DashContext:
        """Gather the image page ids over the KV range for the batch."""
        if idx_mapping.shape[0] == 0 or s_max <= 0:
            return DashContext(img_page_tok=None)
        rows = idx_mapping.long() * PAGE_ROPE_NUM_DIMS + PAGE_ROPE_IMG_CH
        img = self.channels[rows][:, :s_max]
        return DashContext(img_page_tok=img)
