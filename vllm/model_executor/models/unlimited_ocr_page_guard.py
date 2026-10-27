# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PageCountGuard logits processor for Unlimited-OCR-2 multi-page parsing.

In the multi-page layout (``"<PAGE><image>" * n_pages + "Multi page
parsing."``) the model must emit exactly one ``<PAGE>`` marker per prompt
page marker. This request-level guard enforces that contract, mirroring the
reference ``_uocr_page_count_guard``:

* while fewer ``<PAGE>`` markers than prompt pages have been generated, EOS is
  masked so a page cannot be terminated early;
* once the page count is reached EOS is allowed, and generating further
  ``<PAGE>`` markers forces EOS;
* each page is capped at ``max_page_tokens`` generated tokens -- hitting the
  cap forces ``<PAGE>`` (next page) or EOS (last page).

Register it via the CLI::

    --logits-processors \\
      vllm.model_executor.models.unlimited_ocr_page_guard:PageCountGuardPerReqLogitsProcessor

Per-request parameters are read from ``SamplingParams.extra_args``
(``page_token_id``, ``eos_token_id``, ``max_page_tokens``); the token ids fall
back to ``hf_config.sparse_attention_config.page_token_id`` and
``hf_config.eos_token_id`` when not given. The processor is inactive for
requests (or v1 configs) where no page token id can be resolved.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

DEFAULT_MAX_PAGE_TOKENS = 20000


class PageCountGuardLogitsProcessor:
    """Per-request page-count guard applied to generated tokens only."""

    def __init__(
        self,
        page_token_id: int,
        eos_token_id: int,
        max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
    ):
        self.page_token_id = page_token_id
        self.eos_token_id = eos_token_id
        self.max_page_tokens = max_page_tokens
        self.expected_pages: int | None = None
        self.marker_cum = 0
        self.page_len = 0
        self.scanned = 0

    def __call__(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.expected_pages is None:
            self.expected_pages = prompt_ids.count(self.page_token_id)
        if self.expected_pages <= 0:
            return logits

        n = len(output_ids)
        if n > self.scanned:
            tail = output_ids[self.scanned : n]
            num_markers = tail.count(self.page_token_id)
            if num_markers:
                self.marker_cum += num_markers
                self.page_len = tail[::-1].index(self.page_token_id)
            else:
                self.page_len += len(tail)
            self.scanned = n

        if self.marker_cum > self.expected_pages:
            self._force_token(logits, self.eos_token_id)
        elif 0 < self.max_page_tokens <= self.page_len:
            force = (
                self.page_token_id
                if self.marker_cum < self.expected_pages
                else self.eos_token_id
            )
            self._force_token(logits, force)
        elif self.marker_cum < self.expected_pages:
            logits[self.eos_token_id] = -float("inf")
        return logits

    @staticmethod
    def _force_token(logits: torch.Tensor, token_id: int) -> None:
        keep = logits[token_id].clone()
        logits[:] = -float("inf")
        logits[token_id] = keep


class PageCountGuardPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Adapter wiring ``PageCountGuardLogitsProcessor`` into the v1 sampler."""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        hf_config = vllm_config.model_config.hf_config
        sparse_config = getattr(hf_config, "sparse_attention_config", None)
        self._default_page_token_id = self._read(sparse_config, "page_token_id")
        eos_token_id = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        self._default_eos_token_id = eos_token_id

    @staticmethod
    def _read(config: Any, key: str) -> Any:
        if config is None:
            return None
        if isinstance(config, Mapping):
            return config.get(key)
        return getattr(config, key, None)

    @staticmethod
    def _validate_token_id(name: str, value: Any):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"`{name}` has to be a non-negative integer, got {value}.")

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra_args = params.extra_args or {}
        for name in ("page_token_id", "eos_token_id"):
            if (value := extra_args.get(name)) is not None:
                cls._validate_token_id(name, value)
        value = extra_args.get("max_page_tokens")
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            raise ValueError(f"`max_page_tokens` has to be an integer, got {value}.")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        extra_args = params.extra_args or {}
        page_token_id = extra_args.get("page_token_id", self._default_page_token_id)
        if page_token_id is None:
            return None
        self._validate_token_id("page_token_id", page_token_id)

        eos_token_id = extra_args.get("eos_token_id", self._default_eos_token_id)
        if eos_token_id is None:
            raise ValueError(
                "`eos_token_id` is required for the page count guard but could "
                "not be resolved from `extra_args` or the model config."
            )
        self._validate_token_id("eos_token_id", eos_token_id)

        max_page_tokens = extra_args.get("max_page_tokens", DEFAULT_MAX_PAGE_TOKENS)
        if not isinstance(max_page_tokens, int) or isinstance(max_page_tokens, bool):
            raise ValueError(
                f"`max_page_tokens` has to be an integer, got {max_page_tokens}."
            )

        return PageCountGuardLogitsProcessor(
            page_token_id=page_token_id,
            eos_token_id=eos_token_id,
            max_page_tokens=max_page_tokens,
        )


class PageCountGuardV2LogitsProcessor(LogitsProcessor):
    """Model runner V2 version of the page-count guard.

    Same rules as ``PageCountGuardLogitsProcessor`` but driven through the
    batched V2 sampler interface: per-slot state is keyed by the persistent
    request slot and the generated-token tail is re-read from the runner's
    committed token history each step.
    """

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        hf_config = vllm_config.model_config.hf_config
        sparse_config = getattr(hf_config, "sparse_attention_config", None)
        self._default_page_token_id = PageCountGuardPerReqLogitsProcessor._read(
            sparse_config, "page_token_id"
        )
        eos_token_id = getattr(hf_config, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        self._default_eos_token_id = eos_token_id

        self.all_token_ids = req_states.all_token_ids
        self.prompt_len_np = req_states.prompt_len.np
        max_num_reqs = req_states.max_num_reqs
        self.active = np.zeros(max_num_reqs, dtype=bool)
        self.page_token_id = np.zeros(max_num_reqs, dtype=np.int64)
        self.eos_token_id = np.zeros(max_num_reqs, dtype=np.int64)
        self.max_page_tokens = np.zeros(max_num_reqs, dtype=np.int64)
        self.expected_pages = np.full(max_num_reqs, -1, dtype=np.int64)
        self.marker_cum = np.zeros(max_num_reqs, dtype=np.int64)
        self.page_len = np.zeros(max_num_reqs, dtype=np.int64)
        self.scanned = np.zeros(max_num_reqs, dtype=np.int64)

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        PageCountGuardPerReqLogitsProcessor.validate_params(sampling_params)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.active[req_idx] = False
        extra_args = sampling_params.extra_args or {}
        page_token_id = extra_args.get("page_token_id", self._default_page_token_id)
        if page_token_id is None:
            return False
        PageCountGuardPerReqLogitsProcessor._validate_token_id(
            "page_token_id", page_token_id
        )
        eos_token_id = extra_args.get("eos_token_id", self._default_eos_token_id)
        if eos_token_id is None:
            raise ValueError(
                "`eos_token_id` is required for the page count guard but could "
                "not be resolved from `extra_args` or the model config."
            )
        PageCountGuardPerReqLogitsProcessor._validate_token_id(
            "eos_token_id", eos_token_id
        )
        max_page_tokens = extra_args.get("max_page_tokens", DEFAULT_MAX_PAGE_TOKENS)
        if not isinstance(max_page_tokens, int) or isinstance(max_page_tokens, bool):
            raise ValueError(
                f"`max_page_tokens` has to be an integer, got {max_page_tokens}."
            )

        self.active[req_idx] = True
        self.page_token_id[req_idx] = page_token_id
        self.eos_token_id[req_idx] = eos_token_id
        self.max_page_tokens[req_idx] = max_page_tokens
        self.expected_pages[req_idx] = -1  # counted lazily at the first apply
        self.marker_cum[req_idx] = 0
        self.page_len[req_idx] = 0
        self.scanned[req_idx] = 0
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not self.active[ctx.idx_mapping_np].any():
            return logits
        # NOTE(eager): per-row host reads of the committed token tail.
        for row, req in enumerate(ctx.idx_mapping_np):
            if not self.active[req]:
                continue
            page_token_id = int(self.page_token_id[req])
            prompt_len = int(self.prompt_len_np[req])
            expected = int(self.expected_pages[req])
            if expected < 0:
                # First apply for this request: the prompt is now committed,
                # so the page markers can be counted from the token history.
                prompt_ids = self.all_token_ids.gpu[req, :prompt_len].tolist()
                expected = prompt_ids.count(page_token_id)
                self.expected_pages[req] = expected
            if expected <= 0:
                continue
            total_len = int(ctx.seq_lens_upper_bound_np[row])
            output_ids = self.all_token_ids.gpu[req, prompt_len:total_len].tolist()

            n = len(output_ids)
            scanned = int(self.scanned[req])
            if n > scanned:
                tail = output_ids[scanned:]
                num_markers = tail.count(page_token_id)
                if num_markers:
                    self.marker_cum[req] += num_markers
                    self.page_len[req] = tail[::-1].index(page_token_id)
                else:
                    self.page_len[req] += len(tail)
                self.scanned[req] = n

            marker_cum = int(self.marker_cum[req])
            eos_token_id = int(self.eos_token_id[req])
            if marker_cum > expected:
                _force_token(logits[row], eos_token_id)
            elif 0 < self.max_page_tokens[req] <= self.page_len[req]:
                force = page_token_id if marker_cum < expected else eos_token_id
                _force_token(logits[row], force)
            elif marker_cum < expected:
                logits[row, eos_token_id] = -float("inf")
        return logits


def _force_token(logits: torch.Tensor, token_id: int) -> None:
    keep = logits[token_id].clone()
    logits[:] = -float("inf")
    logits[token_id] = keep
