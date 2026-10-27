# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Unlimited-OCR model compatible with HuggingFace weights.

Unlimited-OCR (``baidu/Unlimited-OCR``) shares
the exact DeepSeek-OCR (gundam, ``base_size=1024`` / ``image_size=640`` / crop)
vision stack: a DeepEncoder (SAM-ViT-B + CLIP-L) followed by a linear MLP
projector, with the same image-token tiling layout. The only difference is the
language backbone, which is a DeepSeek-V2 *MoE* (64 routed + 2 shared experts,
``first_k_dense_replace=1``) that uses plain multi-head attention
(``use_mla=False``, ``qk_nope_head_dim == qk_rope_head_dim == 0``) instead of
the dense MLA decoder used by DeepSeek-OCR.

vLLM's ``DeepseekV2DecoderLayer`` already dispatches to the plain-MHA
``DeepseekAttention`` whenever ``qk_nope_head_dim == qk_rope_head_dim == 0`` and
builds the MoE blocks straight from the config, so the whole DeepSeek-OCR
multimodal wrapper can be reused verbatim. Model-specific config (language
backbone architecture, FlexAttention for R-SWA, vision encoder backend, and
``rswa_window``) is applied in ``UnlimitedOCRForCausalLMConfig``.

Attention backend: the reference applies Reference Sliding Window Attention
(R-SWA) -- the prompt/image tokens form a globally-visible prefix while the
*generated* tokens additionally attend only a fixed sliding window (128) of
recent tokens. We reproduce this with backend-specific custom masks: FA4 uses a
``mask_mod``, FlexAttention uses a Triton block mask, and TritonAttention uses a
unified-attention decode mask. FlashInfer's paged decode exposes no custom mask.
The window size is published via ``model_config.rswa_window``, which the model
runner reads to plumb per-request prefix lengths into the attention metadata.

The *vision encoder* (DeepEncoder's CLIP stage, head_dim 64) is unaffected and
does not use R-SWA: it runs a single full-attention prefill pass. FlashAttention,
Triton and torch SDPA all produce correct, equally fast results; only FlashInfer
is incompatible (its ViT path asserts on the varlen cu_seqlens metadata that this
CLIP encoder never builds). We default the encoder to FlashAttention and
transparently fall back to it if FlashInfer is requested.

To suppress repetition on long documents, use ``NGramPerReqLogitsProcessor`` from
this module (same request-level processor as DeepSeek-OCR) with::

    SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args={"ngram_size": 35, "window_size": 128},
    )

Image processing
----------------
Unlimited-OCR supports up to 32 local crops (vs 6 for DeepSeek-OCR), i.e.
``dynamic_preprocess`` runs with ``max_num=32``.

Multi-image requests fall back to non-crop mode: crop ("gundam") mode is only
used for single-image input. DeepSeek-OCR does *not* have this restriction.

Because that fallback makes the per-image processor output depend on *how many*
images are in the request, it breaks the assumption behind vLLM's per-item
multimodal processing cache (``MultiModalProcessorOnlyCache``). We handle this
the same way ``DeepseekVL2MultiModalProcessor`` does: only the single-image case
(which always crops) is cached, while multi-image requests bypass the cache and
are recomputed fresh -- see ``_cached_apply_hf_processor`` below. This keeps the
processing cache consistent (verified by ``test_processing_correctness``).

Unlimited-OCR-2 (``baidu/Unlimited-OCR-2``)
-------------------------------------------
v2 ships the same architecture and model_type; the generations are told apart
by the config: v2 carries ``sparse_attention_config``. There the language
model's attention layers are DASH SR-SWA layers
(``unlimited_ocr2.attention.UnlimitedOCR2Attention``, dispatched in
``DeepseekV2DecoderLayer``) instead of the R-SWA path: a per-layer router
scores 32-token blocks over a paged fp32 index cache, selects a top-K read
set under a same-page image-block constraint, and attends it plus a 128-token
local window with the DASH bias-aware block-sparse Triton kernel
(``vllm/model_executor/kernels/attention/dash``). The page2d RoPE
coordinates are maintained by the runner-side ``Page2dRopeState`` and reach
the layers as extra position channels; the page-aware routing context
arrives via the ``dash_ctx`` forward argument. v2 requires the V2 model
runner and currently runs eager.

The decode recipe needs the Model Runner V2 variants of the two request-level
logits processors::

    --logits_processors \\
      vllm.model_executor.models.unlimited_ocr:NGramV2LogitsProcessor \\
      vllm.model_executor.models.unlimited_ocr_page_guard:PageCountGuardV2LogitsProcessor
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.inputs import MultiModalHashes
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import MultiModalProcessingResult
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    count_tiles,
)
from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)

from .deepseek_ocr import (
    DeepseekOCRDummyInputsBuilder,
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
    NGramPerReqLogitsProcessor,
)
from .utils import WeightsMapper

__all__ = [
    "NGramPerReqLogitsProcessor",
    "NGramV2LogitsProcessor",
    "UnlimitedOCRForCausalLM",
]

# Unlimited-OCR supports up to 32 local crops (vs 6 for DeepSeek-OCR).
_UNLIMITED_OCR_MAX_CROPS = 32


class UnlimitedOCRProcessingInfo(DeepseekOCRProcessingInfo):
    """ProcessingInfo for Unlimited-OCR: same as DeepSeek-OCR but with
    max_crops=32 instead of 6.  The higher crop count allows tiling very large
    document pages into up to 32 640×640 patches (dynamic_preprocess max_num=32).
    """

    def get_hf_config(self):
        from vllm.transformers_utils.configs.unlimited_ocr import UnlimitedOCRConfig

        return self.ctx.get_hf_config(UnlimitedOCRConfig)

    def get_hf_processor(self, **kwargs: object):
        from vllm.transformers_utils.processors.unlimited_ocr import (
            UnlimitedOCRProcessor,
        )

        v1_processor_config = dict(
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
            crop_mode=CROP_MODE,
            strategy="v1",
            max_crops=_UNLIMITED_OCR_MAX_CROPS,
        )
        return self.ctx.get_hf_processor(
            UnlimitedOCRProcessor,
            **{**v1_processor_config, **kwargs},
        )

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        patch_size = 16
        downsample_ratio = 4

        # Honour the caller-supplied `cropping` flag: multi-image callers pass
        # cropping=False to match UnlimitedOCRProcessor.tokenize_with_images.
        if cropping:
            if image_width <= IMAGE_SIZE and image_height <= IMAGE_SIZE:
                crop_ratio = [1, 1]
            else:
                crop_ratio = count_tiles(
                    image_width,
                    image_height,
                    max_num=_UNLIMITED_OCR_MAX_CROPS,
                    image_size=IMAGE_SIZE,
                )
            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((BASE_SIZE // patch_size) / downsample_ratio)
        h2 = w2 = math.ceil((IMAGE_SIZE // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        # A vertical grid maximizes both crop count and newline tokens.
        return ImageSize(width=IMAGE_SIZE, height=IMAGE_SIZE * _UNLIMITED_OCR_MAX_CROPS)


class UnlimitedOCRMultiModalProcessor(DeepseekOCRMultiModalProcessor):
    """Multimodal processor for Unlimited-OCR.

    Disables crop mode for multi-image requests (to stay consistent with
    ``UnlimitedOCRProcessor.tokenize_with_images``), and -- since that makes the
    per-image output depend on the request's image count -- bypasses the
    per-item processing cache for multi-image requests, exactly like
    ``DeepseekVL2MultiModalProcessor``.

    DeepSeek-OCR does *not* apply either of these.
    """

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        mm_kwargs = self.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
        crop_mode = mm_kwargs.get("crop_mode", CROP_MODE)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_unlimited_ocr(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                assert isinstance(images, ImageProcessorItems)
                size = images.get_image_size(item_idx)

                # Disable crop mode for multi-image input.
                # UnlimitedOCRProcessor.tokenize_with_images applies the same
                # fallback, so both paths must agree on the effective crop flag.
                effective_cropping = bool(crop_mode) and len(images) == 1

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=size.width,
                    image_height=size.height,
                    cropping=effective_cropping,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_unlimited_ocr,
            )
        ]

    def _get_mm_hashes(self, inputs: ProcessorInputs) -> MultiModalHashes:
        hashes = dict(super()._get_mm_hashes(inputs))
        images = inputs.mm_data_items.get("image")
        if isinstance(images, ImageProcessorItems):
            mm_kwargs = self.info.ctx.get_merged_mm_kwargs(
                inputs.hf_processor_mm_kwargs
            )
            cropping = len(images) == 1 and bool(mm_kwargs.get("crop_mode", CROP_MODE))
            algorithm = self.info.ctx.get_mm_config().mm_hasher_algorithm
            hashes["image"] = [
                MultiModalHasher.hash_kwargs(
                    algorithm, mm_hash=mm_hash, cropping=cropping
                )
                for mm_hash in hashes["image"]
            ]
        return hashes

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalProcessingResult:
        # The processor logic differs for single-image (crop) vs multi-image
        # (no crop) requests. The processing cache assumes per-item output is
        # invariant of how many images are passed per prompt, so we only cache
        # the single-image case and recompute multi-image requests fresh.
        if inputs.mm_data_items.get_count("image", strict=False) > 1:
            return self._apply_hf_processor(inputs, timing_ctx)

        return super()._cached_apply_hf_processor(inputs, timing_ctx)


@MULTIMODAL_REGISTRY.register_processor(
    UnlimitedOCRMultiModalProcessor,
    info=UnlimitedOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class UnlimitedOCRForCausalLM(DeepseekOCRForCausalLM):
    # v2 checkpoints (baidu/Unlimited-OCR-2) additionally ship a persistent
    # ``vision_model.embeddings.position_ids`` buffer that the vLLM CLIP
    # embeddings hold as a non-persistent buffer; drop it on load.
    hf_to_vllm_mapper = DeepseekOCRForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_substr={".position_ids": None}
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # v2 (SR-SWA/DASH) is detected purely from the config: both
        # generations share model_type/architectures.
        self.is_v2 = getattr(self.config, "sparse_attention_config", None) is not None
        if self.is_v2:
            from .unlimited_ocr2.attention import (
                PAGE_ROPE_FLAT_CH,
                DashContextProvider,
                UnlimitedOCR2Attention,
            )

            self._page_rope_flat_ch = PAGE_ROPE_FLAT_CH
            self._dash_ctx_provider = DashContextProvider()
            for module in self.modules():
                if isinstance(module, UnlimitedOCR2Attention):
                    module.dash_ctx_provider = self._dash_ctx_provider

    def get_model_state_cls(self):
        """Model runner V2 ModelState: v2 needs the page2d rope state."""
        if self.is_v2:
            from vllm.v1.worker.gpu.model_states.unlimited_ocr2 import (
                UnlimitedOCR2ModelState,
            )

            return UnlimitedOCR2ModelState

        from vllm.v1.worker.gpu.model_states.default import DefaultModelState

        return DefaultModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        dash_ctx=None,
        **kwargs: object,
    ):
        # dash_ctx carries the per-KV-token image page ids for the DASH
        # page-aware routing (None for v1 and for dummy/profile runs).
        if self.is_v2:
            self._dash_ctx_provider.dash_ctx = dash_ctx
            # Stash the page2d channels for the DASH layers; the language
            # model itself only understands flat 1-D positions.
            self._dash_ctx_provider.positions = positions
            positions = positions[self._page_rope_flat_ch]
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )


class NGramV2LogitsProcessor(LogitsProcessor):
    """Model runner V2 version of the no-repeat-n-gram processor.

    Same ban rule as ``NoRepeatNGramLogitsProcessor`` (ban the token that
    would repeat an n-gram whose (n-1)-prefix matches the generated tail,
    searched over a trailing window of the generated tokens), driven through
    the batched V2 sampler interface.
    """

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        self.all_token_ids = req_states.all_token_ids
        self.prompt_len_np = req_states.prompt_len.np
        max_num_reqs = req_states.max_num_reqs
        self.active = np.zeros(max_num_reqs, dtype=bool)
        self.ngram_size = np.zeros(max_num_reqs, dtype=np.int64)
        self.window_size = np.zeros(max_num_reqs, dtype=np.int64)
        self.whitelist: list[set[int]] = [set() for _ in range(max_num_reqs)]

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        NGramPerReqLogitsProcessor.validate_params(sampling_params)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.active[req_idx] = False
        self.whitelist[req_idx] = set()
        extra_args = sampling_params.extra_args or {}
        ngram_size = extra_args.get("ngram_size")
        if ngram_size is None:
            return False
        window_size = extra_args.get("window_size", 100)
        assert isinstance(ngram_size, int) and isinstance(window_size, int)
        whitelist_token_ids = extra_args.get("whitelist_token_ids", None)

        self.active[req_idx] = True
        self.ngram_size[req_idx] = ngram_size
        self.window_size[req_idx] = window_size
        self.whitelist[req_idx] = (
            set(whitelist_token_ids) if whitelist_token_ids else set()
        )
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not self.active[ctx.idx_mapping_np].any():
            return logits
        # NOTE(eager): per-row host reads of the committed token tail.
        for row, req in enumerate(ctx.idx_mapping_np):
            if not self.active[req]:
                continue
            ngram_size = int(self.ngram_size[req])
            prompt_len = int(self.prompt_len_np[req])
            total_len = int(ctx.seq_lens_upper_bound_np[row])
            num_output = total_len - prompt_len
            if num_output < ngram_size:
                continue
            output_ids = self.all_token_ids.gpu[req, prompt_len:total_len].tolist()

            current_prefix = tuple(output_ids[-(ngram_size - 1) :])
            window_size = int(self.window_size[req])
            search_start = max(0, num_output - window_size)
            search_end = num_output - ngram_size + 1
            banned_tokens = set()
            for i in range(search_start, search_end):
                ngram = tuple(output_ids[i : i + ngram_size])
                if ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])
            banned_tokens -= self.whitelist[req]
            if banned_tokens:
                logits[row, list(banned_tokens)] = -float("inf")
        return logits
