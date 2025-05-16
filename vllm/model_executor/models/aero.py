# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

import torch
from transformers import ProcessorMixin

from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioDummyInputsBuilder, Qwen2AudioForConditionalGeneration,
    Qwen2AudioMultiModalProcessor, Qwen2AudioProcessingInfo,
    _get_feat_extract_output_lengths)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)

from .utils import AutoWeightsLoader, WeightsMapper


class AeroProcessingInfo(Qwen2AudioProcessingInfo):

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> ProcessorMixin:
        return self.ctx.get_hf_processor(**kwargs)


class AeroDummyInputsBuilder(Qwen2AudioDummyInputsBuilder[AeroProcessingInfo]):
    pass


class AeroMultiModalProcessor(Qwen2AudioMultiModalProcessor[AeroProcessingInfo]
                              ):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")

        audio_token_id = vocab[audio_token]

        feature_attention_mask = out_mm_kwargs.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))

            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_aero(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(f"The audio (len={audio_len}) is too short "
                                 "to be represented inside the model")

            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_aero,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(AeroMultiModalProcessor,
                                        info=AeroProcessingInfo,
                                        dummy_inputs=AeroDummyInputsBuilder)
class AeroForConditionalGeneration(Qwen2AudioForConditionalGeneration):

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "audio_modal_projector.": "multi_modal_projector.",
            })
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
