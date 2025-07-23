from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypedDict, Union

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig
from transformers.models.qwen2_audio import (Qwen2AudioConfig,
                                             Qwen2AudioEncoder,
                                             Qwen2AudioProcessor)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig, QuantizationConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

from .qwen2_audio import Qwen2AudioForConditionalGeneration, Qwen2AudioMultiModalProcessor, Qwen2AudioDummyInputsBuilder, Qwen2AudioProcessingInfo


class VoxtralProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": 5}  # Performance tends to degrade after 5

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor


class VoxtralDummyInputsBuilder(BaseDummyInputsBuilder[VoxtralProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class VoxtralMultiModalProcessor(BaseMultiModalProcessor[VoxtralProcessingInfo]
                                 ):

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        audio_id = processor.audio_token_id

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)

            nb_audio_tokens = processor.get_num_audio_tokens(audio_len)

            return [audio_id] * nb_audio_tokens

        return [
            PromptReplacement(
                modality="audio",
                target="",  # Never match the prompt (see below note)
                replacement=get_replacement,
            ),
        ]

    def _get_data_parser(self) -> MultiModalDataParser:
        sampling_rate = self.info.get_hf_processor().sampling_rate
        return MultiModalDataParser(target_sr=sampling_rate)


class VoxtralMultiModalProjector(nn.Module):
    def __init__(self, text_hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear_1 = ReplicatedLinear(intermediate_size, text_hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear_2 = ReplicatedLinear(text_hidden_size, text_hidden_size, bias=False)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralMultiModalProcessor,
    info=VoxtralProcessingInfo,
    dummy_inputs=VoxtralDummyInputsBuilder)
class VoxtralForConditionalGeneration(Qwen2AudioForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        print(vllm_config.model_config.hf_config)
        vllm_config.model_config.hf_config.text_config.architectures = ["LlamaForCausalLM"]
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )
        print(self)
    
    def _init_multi_modal_projector(self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        prefix: str,
    ):
        return VoxtralMultiModalProjector(
            text_hidden_size=config.text_config.hidden_size,
            intermediate_size=config.audio_config.intermediate_size,
        )