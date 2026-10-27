# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-side components of baidu/Unlimited-OCR-2 SR-SWA attention.

Framework-agnostic ports from the HF reference implementation:

- :mod:`.dash_router`: DASH router (1.5-entmax, ``head_cls`` block summary,
  top-M entmax window + hard top-K read set with mean-centered log-prior
  bias).
- :mod:`.page2d_rope`: page2d dual-axis RoPE, page-coordinate construction,
  and the page-aware routing candidate mask.
- :mod:`.page_state`: batched incremental page2d decode state machine.
"""

from .attention import (
    PAGE_ROPE_NUM_DIMS,
    DashAttention,
    DashContext,
    DashIndexCache,
    UnlimitedOCR2Attention,
)
from .dash_router import (
    DashRouteParams,
    DashRouter,
    GemmaRMSNorm,
    apply_rotary_pos_emb,
    entmax15,
)
from .page2d_rope import (
    Page2DRotaryEmbedding,
    build_image_block_mask,
    build_page_aware_ids,
    build_page_candidate_mask,
    build_page_rope_positions,
)
from .page_state import PageRopeState

__all__ = [
    "PAGE_ROPE_NUM_DIMS",
    "DashAttention",
    "DashContext",
    "DashIndexCache",
    "DashRouteParams",
    "DashRouter",
    "GemmaRMSNorm",
    "Page2DRotaryEmbedding",
    "PageRopeState",
    "UnlimitedOCR2Attention",
    "apply_rotary_pos_emb",
    "build_image_block_mask",
    "build_page_aware_ids",
    "build_page_candidate_mask",
    "build_page_rope_positions",
    "entmax15",
]
