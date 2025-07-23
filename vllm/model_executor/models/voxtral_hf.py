from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, TypedDict, Union

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.qwen2_audio import (Qwen2AudioConfig,
                                             Qwen2AudioEncoder,
                                             Qwen2AudioProcessor)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import VllmConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
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


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2AudioMultiModalProcessor,
    info=Qwen2AudioProcessingInfo,
    dummy_inputs=Qwen2AudioDummyInputsBuilder)
class VoxtralForConditionalGeneration(Qwen2AudioForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        vllm_config.model_config.hf_config.text_config.architectures = ["LlamaForCausalLM"]
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
        )