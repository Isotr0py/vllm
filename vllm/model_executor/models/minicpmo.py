# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only MiniCPM-O model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping, Sequence
from typing import (Any, Callable, Dict, Literal, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
from torch import nn
from transformers import BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.whisper.modeling_whisper import (
    ACT2FN, WHISPER_ATTENTION_CLASSES, WhisperConfig, WhisperEncoder)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.multimodal.parse import (AudioItem, AudioProcessorItems,
                                   DictEmbeddingItems, ModalityData,
                                   ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.multimodal.profiling import ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.utils import flatten_2d_lists

from .minicpmv import (MiniCPMV2_6, MiniCPMVDummyInputsBuilder,
                       MiniCPMVMultiModalDataParser,
                       MiniCPMVMultiModalProcessor, MiniCPMVProcessingInfo,
                       _minicpmv_field_config)
from .utils import (AutoWeightsLoader, cast_overflow_tensors, flatten_bn,
                    maybe_prefix)

CPU_DEVICE = torch.device("cpu")


class MiniCPMOAudioFeatureInputs(TypedDict):
    type: Literal["audio_features"]
    audio_features: torch.Tensor
    """
    Shape: `(batch_size * num_audios * num_slices, num_channels, length)`
    Slice here means chunk. Audio that is too long will be split into slices,
    which is the same as image.
    Padding is used therefore `audio_features` is `torch.Tensor`.
    """

    audio_feature_lens: torch.Tensor
    """
    Shape: `(batch_size * num_audios * num_slices)`

    This should be feature length of each audio slice, 
    which equals to `audio_features.shape[-1]`
    """

    audio_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_audios * num_slices, 2)`

    This should be in `(start, stop)` format.
    """


class MiniCPMOAudioEmbeddingInputs(TypedDict):
    type: Literal["audio_embeds"]
    audio_embeds: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_slices, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    instead of a batched tensor.
    Length of each slice may vary, so pass it as a list.
    """
    audio_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_audios * num_slices, 2)`

    This should be in `(start, stop)` format.
    """


MiniCPMOAudioInputs = Union[MiniCPMOAudioFeatureInputs,
                            MiniCPMOAudioEmbeddingInputs]


def _minicpmo_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        **_minicpmv_field_config(hf_inputs),
        audio_features=MultiModalFieldConfig.batched("audio"),
        audio_feature_lens=MultiModalFieldConfig.batched("audio"),
        audio_embeds=MultiModalFieldConfig.batched("audio"),
    )


class MiniCPMOAudioEmbeddingItems(DictEmbeddingItems):

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        super().__init__(
            data,
            modality="image",
            required_fields={"audio_embeds"},
            fields_factory=fields_factory,
        )


class MiniCPMOMultiModalDataParser(MiniCPMVMultiModalDataParser):

    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[AudioItem]],
    ) -> ModalityDataItems[Any, Any]:
        if isinstance(data, dict):
            return MiniCPMOAudioEmbeddingItems(
                data,
                fields_factory=_minicpmo_field_config,
            )

        return super()._parse_audio_data(data)


class MiniCPMOProcessingInfo(MiniCPMVProcessingInfo):
    audio_pattern = "(<audio>./</audio>)"

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "video": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "audio": self.get_max_audio_tokens(),
            "video": self.get_max_video_tokens(seq_len),
        }

    def get_default_audio_pool_step(self) -> int:
        return 2

    def get_default_audio_sampling_rate(self) -> int:
        return 16000

    def get_chunk_length(self) -> int:
        return self.get_hf_config().audio_chunk_length

    def get_max_audio_tokens_per_chunk(self) -> int:
        pool_step = self.get_default_audio_pool_step()
        fbank_feat_in_chunk = 100
        cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
        num_audio_tokens = (cnn_feat_in_chunk - pool_step) // pool_step + 1
        return num_audio_tokens + 2  # <audio>(<unk>*N)</audio>

    def get_max_audio_chunks_with_most_features(self) -> int:
        return 30

    def get_max_audio_tokens(self) -> int:
        return self.get_max_audio_tokens_per_chunk(
        ) * self.get_max_audio_chunks_with_most_features()

    def get_audio_len_by_num_chunks(self, num_chunks: int) -> int:
        sampling_rate = self.get_default_audio_sampling_rate()
        # exclude <audio> </audio>
        num_tokens_per_chunk = self.get_max_audio_tokens_per_chunk() - 2
        return int(num_chunks * sampling_rate / num_tokens_per_chunk) + 1

    def get_num_frames_with_most_features(self, seq_len: int) -> int:
        mm_config = self.ctx.get_mm_config()
        max_images = mm_config.get_limit_per_prompt("image")
        max_videos = mm_config.get_limit_per_prompt("video")
        max_audios = mm_config.get_limit_per_prompt("audio")

        # count <image_idx></image_idx> tokens
        # which are not in get_max_image_tokens
        max_image_tokens = self.get_max_image_tokens(
        ) * max_images + 4 * max_images
        max_audio_tokens = self.get_max_audio_tokens(
        ) * max_audios + 2 * max_audios
        max_total_frames = self.get_max_video_frames(seq_len -
                                                     max_image_tokens -
                                                     max_audio_tokens)

        num_frames = max(max_total_frames // max(max_videos, 1), 1)

        return num_frames


class MiniCPMODummyInputsBuilder(
        MiniCPMVDummyInputsBuilder[MiniCPMOProcessingInfo]):

    def get_dummy_processor_inputs(
            self, seq_len: int, mm_counts: Mapping[str,
                                                   int]) -> ProcessorInputs:
        num_audios = mm_counts.get("audio", 0)
        audio_len = self.info.get_max_audio_chunks_with_most_features() * \
            self.info.get_default_audio_sampling_rate()

        processor_inputs = super().get_dummy_processor_inputs(
            seq_len, mm_counts)
        mm_data = {
            "image":
            processor_inputs.mm_data["image"],
            "video":
            processor_inputs.mm_data["video"],
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }

        audio_prompt_texts = self.info.audio_pattern * num_audios

        return ProcessorInputs(prompt_text=processor_inputs.prompt_text + \
                               audio_prompt_texts,
                               mm_data=mm_data)


class MiniCPMOMultiModalProcessor(
        MiniCPMVMultiModalProcessor[MiniCPMOProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MiniCPMOMultiModalDataParser(
            target_sr=self.info.get_default_audio_sampling_rate())

    def get_audio_prompt_texts(self,
                               audio_lens: int,
                               chunk_input: bool = True,
                               chunk_length: int = 1) -> str:
        return self.info.get_hf_processor().get_audio_placeholder(
            audio_lens, chunk_input, chunk_length)

    def get_special_tokens(self) -> Dict[str, torch.Tensor]:
        tokenizer = self.info.get_tokenizer()
        special_tokens = super().get_special_tokens()
        if hasattr(tokenizer, "audio_start_id"):
            special_tokens["audio_start_id"] = torch.tensor(
                tokenizer.audio_start_id)
            special_tokens["audio_end_id"] = torch.tensor(
                tokenizer.audio_end_id)
        return special_tokens

    def process_audios(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (audios := mm_data.get("audios")) is None:
            return {}

        parsed_audios = (self._get_data_parser().parse_mm_data({
            "audio": audios
        }).get_items("audio", AudioProcessorItems))

        audio_inputs = self._base_call_hf_processor(
            prompts=[self.info.audio_pattern] * len(parsed_audios),
            mm_data={"audios": [[audio] for audio in parsed_audios]},
            mm_kwargs={
                **mm_kwargs, "chunk_input": True
            },
            out_keys={"audio_features", "audio_feature_lens"},
        )

        # Avoid padding since we need the output for each audio to be
        # independent of other audios for the cache to work correctly
        unpadded_audio_features = [
            feat[:, :feature_len] for feat, feature_len in zip(
                audio_inputs["audio_features"],
                audio_inputs["audio_feature_lens"],
            )
        ]
        audio_inputs["audio_features"] = unpadded_audio_features

        return audio_inputs

    def get_placeholder_match_pattern(self) -> str:
        return r"\(<(image|video|audio)>./</\1>\)"

    def process_mm_inputs(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        return {
            **super().process_mm_inputs(mm_data, mm_kwargs),
            **self.process_audios(mm_data, mm_kwargs),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        base_updates = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        audio_placeholder = self.info.audio_pattern

        def get_audio_replacement(item_idx: int):
            audios = mm_items.get_items(
                "audio", (MiniCPMOAudioEmbeddingItems, AudioProcessorItems))

            if isinstance(audios, MiniCPMOAudioEmbeddingItems):
                single_audio_embeds = audios.get(item_idx)["audio_embeds"]
                audio_len = self.info.get_audio_len_by_num_chunks(
                    sum(chunk_embeds.shape[0]
                        for chunk_embeds in single_audio_embeds))
            else:
                audio_len = audios.get_audio_length(item_idx)

            return self.get_audio_prompt_texts(audio_len)

        return [
            *base_updates,
            PromptReplacement(modality="audio",
                              target=audio_placeholder,
                              replacement=get_audio_replacement),
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _minicpmo_field_config(hf_inputs)


class MultiModalProjector(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim,
                                 out_features=out_dim,
                                 bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim,
                                 out_features=out_dim,
                                 bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMWhisperEncoderLayer(nn.Module):

    def __init__(self, config: WhisperConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WHISPER_ATTENTION_CLASSES[
            config._attn_implementation](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                config=config,
                layer_idx=layer_idx,
            )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        past_key_values = None
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
        )
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.activation_dropout,
                                              training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        outputs = (hidden_states, )

        return outputs


class MiniCPMWhisperEncoder(WhisperEncoder):

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            MiniCPMWhisperEncoderLayer(config, layer_idx=i)
            for i in range(config.encoder_layers)
        ])

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype,
                                           device=self.conv1.weight.device)

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight

        embed_pos = embed_pos[:inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)

        encoder_states = ()

        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states, )
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                )

                hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        encoder_states = encoder_states + (hidden_states, )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
        )


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOMultiModalProcessor,
    info=MiniCPMOProcessingInfo,
    dummy_inputs=MiniCPMODummyInputsBuilder)
class MiniCPMO(MiniCPMV2_6):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.apm = self.init_audio_module(vllm_config=vllm_config,
                                          prefix=maybe_prefix(prefix, "apm"))

    def init_audio_module(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Do not use parameters temporarily
        audio_config = self.config.audio_config
        model = MiniCPMWhisperEncoder(audio_config)
        audio_output_dim = int(audio_config.encoder_ffn_dim // 4)
        self.audio_avg_pooler = \
            nn.AvgPool1d(self.config.audio_pool_step,
                         stride=self.config.audio_pool_step)
        self.audio_projection_layer = \
            MultiModalProjector(in_dim=audio_output_dim,out_dim=self.embed_dim)
        self.audio_encoder_layer = -1
        return model

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["tts"])
        return loader.load_weights(weights)

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = CPU_DEVICE,
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size,
                            0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead,
                         size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self,
                                         input_lengths: torch.LongTensor):
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn -
            self.config.audio_pool_step) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(
            dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    # Copied from HF repo of MiniCPM-o-2_6,
    # designed for batched inputs and outputs
    def get_audio_hidden_states(self, data: MiniCPMOAudioInputs,
                                chunk_length: int) -> list[torch.Tensor]:
        wavforms = data.get(
            "audio_features",
            [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = [data.get("audio_feature_lens",
                                           [])]  # list, [[x1, x2], [y1], [z1]]

        if len(wavforms) == 0:
            return []

        audio_feature_lens = torch.hstack(audio_feature_lens_raw)
        batch_size, _, max_mel_seq_len = wavforms.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1

        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (torch.arange(
            0,
            max_seq_len,
            dtype=audio_feature_lens.dtype,
            device=audio_feature_lens.device).unsqueeze(0).expand(
                batch_size, max_seq_len))
        lengths_expand = audio_feature_lens.unsqueeze(1).expand(
            batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand  # 1 for padded values

        audio_attention_mask_ = padding_mask.view(
            batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len,
                                                  max_seq_len)
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.apm.conv1.weight.dtype,
            device=self.apm.conv1.weight.device)

        if chunk_length > 0:
            chunk_num_frame = int(chunk_length * 50)
            chunk_mask = self.subsequent_chunk_mask(
                size=max_seq_len,
                chunk_size=chunk_num_frame,
                num_left_chunks=-1,
                device=audio_attention_mask_.device,
            )
            audio_attention_mask_ = torch.logical_or(
                audio_attention_mask_, torch.logical_not(chunk_mask))

        audio_attention_mask[audio_attention_mask_] = float("-inf")
        audio_states = self.apm(
            wavforms, attention_mask=audio_attention_mask).hidden_states[
                self.audio_encoder_layer]
        audio_embeds = self.audio_projection_layer(audio_states)

        audio_embeds = audio_embeds.transpose(1, 2)
        audio_embeds = self.audio_avg_pooler(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)

        _, feature_lens_after_pooling = \
            self._get_feat_extract_output_lengths(audio_feature_lens)

        num_audio_tokens = feature_lens_after_pooling

        final_audio_embeds = []
        idx = 0
        for i in range(len(audio_feature_lens_raw)):
            target_audio_embeds = []
            for _ in range(len(audio_feature_lens_raw[i])):
                target_audio_embeds.append(
                    audio_embeds[idx, :num_audio_tokens[idx], :])
                idx += 1
            final_audio_embeds.append(target_audio_embeds)
        return final_audio_embeds

    def get_embedding_with_audios(self, vlm_embedding: torch.Tensor,
                                  audio_inputs: MiniCPMOAudioInputs,
                                  chunk_length: int) -> torch.Tensor:
        device, dtype = vlm_embedding.device, vlm_embedding.dtype
        if audio_inputs["type"] == "audio_embeds":
            audio_embeddings = [
                item.to(device=device, dtype=dtype)
                for item in audio_inputs["audio_embeds"]
            ]
        else:
            audio_embeddings = self.get_audio_hidden_states(
                audio_inputs, chunk_length)[0]
        if audio_embeddings is None or len(audio_embeddings) == 0:
            return vlm_embedding
        audio_bounds = audio_inputs["audio_bounds"]
        if self.config.chunk_input:
            audio_embs = torch.cat(audio_embeddings, dim=0).to(device=device,
                                                               dtype=dtype)
            audio_start_pos = 0
            for bound in audio_bounds:
                audio_len = bound[1] - bound[0]
                vlm_embedding[bound[0]:bound[1]] = audio_embs[
                    audio_start_pos:audio_start_pos + audio_len, :]
                audio_start_pos += audio_len
        else:
            for embs, bound in zip(audio_embeddings, audio_bounds):
                audio_indices = torch.arange(bound[0],
                                             bound[1],
                                             dtype=torch.long).to(device)

                if embs.shape[0] != len(audio_indices):
                    raise ValueError(
                        "Shape mismatch: Trying to assign embeddings "
                        f"of shape {embs.shape} "
                        f"to input indices of length {len(audio_indices)}")
                vlm_embedding[audio_indices] = embs.to(dtype)
        return vlm_embedding

    def _get_audio_bounds(self, input_ids: torch.Tensor,
                          audio_start_id: torch.Tensor,
                          audio_end_id: torch.Tensor) -> torch.Tensor:
        audio_start_tokens, = torch.where(input_ids == audio_start_id[0])
        audio_start_tokens += 1
        audio_end_tokens, = torch.where(input_ids == audio_end_id[0])
        valid_audio_nums = max(len(audio_start_tokens), len(audio_end_tokens))
        return torch.hstack([
            audio_start_tokens[:valid_audio_nums].unsqueeze(-1),
            audio_end_tokens[:valid_audio_nums].unsqueeze(-1)
        ])

    def _parse_and_validate_audio_inputs(
            self, input_ids: torch.Tensor,
            **kwargs: object) -> Optional[MiniCPMOAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        audio_start_id = kwargs.pop("audio_start_id")
        if not isinstance(audio_start_id, torch.Tensor):
            raise ValueError("Incorrect type of audio_start_id. "
                             f"Got type: {type(audio_start_id)}")

        audio_end_id = kwargs.pop("audio_end_id")
        if not isinstance(audio_end_id, torch.Tensor):
            raise ValueError("Incorrect type of audio_end_id. "
                             f"Got type: {type(audio_end_id)}")

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio_embeds. "
                                 f"Got type: {type(audio_embeds)}")

            return MiniCPMOAudioEmbeddingInputs(
                type="audio_embeds",
                audio_embeds=flatten_bn(flatten_2d_lists(audio_embeds),
                                        concat=True),
                audio_bounds=self._get_audio_bounds(input_ids, audio_start_id,
                                                    audio_end_id),
            )

        if audio_features is not None:
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio_features. "
                                 f"Got type: {type(audio_features)}")

            audio_feature_lens = kwargs.pop("audio_feature_lens")
            if not isinstance(audio_feature_lens, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio_feature_lens. "
                                 f"Got type: {type(audio_feature_lens)}")

            return MiniCPMOAudioFeatureInputs(
                type="audio_features",
                audio_features=flatten_bn(audio_features, concat=True),
                audio_feature_lens=flatten_bn(
                    flatten_2d_lists(audio_feature_lens), concat=True),
                audio_bounds=self._get_audio_bounds(input_ids, audio_start_id,
                                                    audio_end_id),
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_inputs(self, input_ids: torch.Tensor,
                                   **kwargs: object):
        image_inputs = self._parse_and_validate_image_inputs(
            input_ids, **kwargs)
        if not any("audio" in key for key in kwargs):
            return image_inputs, None
        audio_inputs = self._parse_and_validate_audio_inputs(
            input_ids, **kwargs)
        return image_inputs, audio_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            vlm_embeddings = None
        else:
            image_inputs, audio_inputs = \
                self._parse_and_validate_inputs(input_ids, **kwargs)
            vlm_embeddings = self.get_embedding_with_vision(
                input_ids, image_inputs)

            if audio_inputs is not None:
                vlm_embeddings = self.get_embedding_with_audios(
                    vlm_embeddings, audio_inputs,
                    self.config.audio_chunk_length)

        # always pass the input via `inputs_embeds`
        # to make sure the computation graph is consistent
        # for `torch.compile` integration
        input_ids = None

        output = self.llm.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=vlm_embeddings,
        )
        return output
