# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model runner state for baidu/Unlimited-OCR-2 (DASH SR-SWA).

Wires the page2d rope state into the V2 GPU model runner: prefill channels
are seeded per request at ``add_request``, decode coordinates advance in
``prepare_inputs`` (once per step, before the forward), and the page-routing
context (per-KV-token image page ids) is handed to the model as the
``dash_ctx`` forward kwarg.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.page_rope import Page2dRopeState
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState


class UnlimitedOCR2ModelState(DefaultModelState):
    """DefaultModelState with the DASH page2d rope state and routing ctx."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        sp = self.model_config.hf_config.sparse_attention_config
        assert sp is not None, "UnlimitedOCR2ModelState requires a v2 config"
        self.rope_state = Page2dRopeState(
            image_token_id=sp.image_token_id,
            page_token_id=sp.page_token_id,
            per_page_image_tokens=sp.per_page_image_tokens,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=device,
        )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        from typing import cast

        rope_state = cast(Page2dRopeState, self.rope_state)
        rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
            input_batch.input_ids,
            input_batch.num_tokens,
        )
        positions = rope_state.get_positions(input_batch.num_tokens_after_padding)
        s_max = int(input_batch.seq_lens_cpu_upper_bound[: input_batch.num_reqs].max())
        dash_ctx = rope_state.build_dash_context(input_batch.idx_mapping, s_max)
        return {"positions": positions, "dash_ctx": dash_ctx}
