# coding=utf-8
# adapted from 
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Idefics3 model."""

import re
from functools import cached_property, partial
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import Idefics3Config

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.models.idefics2_vision_model import Idefics2VisionTransformer
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from .clip import (dummy_image_for_clip, dummy_seq_data_for_clip,
                   get_clip_num_patches)
from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, flatten_bn,
                    merge_multimodal_embeddings, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)


class Idefics3ImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape:
    `(batch_size * num_images, 1+num_patches, num_channels, height, width)`
    """



class Idefics3ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(self, config: Idefics3Config, multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        self.vision_config = config.vision_config
        self.text_config = config.text_config
        self.multimodal_config = multimodal_config

        self.vision_model = Idefics2VisionTransformer(self.vision_config)
        self.text_model = LlamaModel(self.text_config, 
                                     cache_config=cache_config,
                                     quant_config=quant_config)
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.text_model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.text_config.vocab_size,
                                            config.text_config.hidden_size,
                                            quant_config=quant_config)
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()
        
        self.make_empty_intermediate_tensors = (
            self.text_model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[SamplerOutput, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            image_input = self._parse_and_validate_image_input(**kwargs)
            if image_input is not None:
                inputs_embeds = self.text_model.get_input_embeddings(
                    input_ids)
                vision_embeddings = self._process_image_input(image_input)
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.img_context_token_id)
                input_ids = None
            else:
                inputs_embeds = None

        hidden_states = self.text_model(input_ids,
                                        positions,
                                        kv_caches,
                                        attn_metadata,
                                        intermediate_tensors,
                                        inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)