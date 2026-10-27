# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched incremental page2d RoPE state for baidu/Unlimited-OCR-2 decode.

Port of the decode branch of the HF reference ``_main_rope_coords``: after
prefill, every request carries its last token's (tok, page) coordinate and the
per-page prefill token counts. Each decode step looks at the new token id: a
``<PAGE>`` marker flips the request to the next page and resets tok to that
page's prefill length (so the response text of page k continues where page k's
image run ended); any other token just increments tok.

The state is one row per request, advanced with tensor ops only (no data-
dependent host sync), so it can be updated inside a CUDA graph: keep the
tensors in fixed buffers and call :meth:`PageRopeState.advance` with the
step's token ids.
"""

from dataclasses import dataclass

import torch


@dataclass
class PageRopeState:
    """Per-request page2d coordinates for incremental decode.

    Attributes:
        page: [B] long, page id of each request's last token.
        tok: [B] long, intra-page offset of each request's last token.
        pf_cnt: [B, n_pg] long, prefill token count per page bucket
            (bucket 0 = reserved no-page region).
        n_pg: number of page buckets (max page id + 1).

    """

    page: torch.Tensor
    tok: torch.Tensor
    pf_cnt: torch.Tensor
    n_pg: int

    @classmethod
    def from_prefill(
        cls,
        tok_pos: torch.Tensor,
        page_pos: torch.Tensor,
        pf_cnt: torch.Tensor,
        n_pg: int,
    ) -> "PageRopeState":
        """Seed the decode state from the prefill coordinates.

        Args:
            tok_pos: [B, S] long intra-page offsets from
                ``build_page_rope_positions``.
            page_pos: [B, S] long page ids from ``build_page_rope_positions``.
            pf_cnt: [B, n_pg] long per-page prefill counts.
            n_pg: number of page buckets.

        """
        return cls(
            page=page_pos[:, -1].clone(),
            tok=tok_pos[:, -1].clone(),
            pf_cnt=pf_cnt.clone(),
            n_pg=int(n_pg),
        )

    @torch.no_grad()
    def advance(
        self,
        input_ids: torch.Tensor,
        page_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one decode step and return the new tokens' coordinates.

        The state tensors are updated in place, so callers can allocate them
        once and replay under CUDA graphs.

        Args:
            input_ids: [B, Lq] token ids of the current step (only the last
                column is consumed; Lq is 1 for plain decode and the chunk
                tail for speculative verify).
            page_token_id: id of the ``<PAGE>`` marker token.

        Returns:
            (tok_pos [B, 1] long, page_pos [B, 1] long) for the new token.

        """
        new = input_ids[:, -1].to(self.page.device)
        is_pg = new == page_token_id
        page = torch.where(is_pg, (self.page + 1).clamp(max=self.n_pg - 1), self.page)
        pg_start = self.pf_cnt.gather(1, page[:, None].clamp(min=0)).squeeze(1)
        tok = torch.where(is_pg, pg_start, self.tok + 1)
        self.page.copy_(page)
        self.tok.copy_(tok)
        return self.tok[:, None], self.page[:, None]
