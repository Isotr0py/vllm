# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3_5RMSNorm,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # GDN
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            # self attention
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # mlp
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        is_fused_expert = False
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )
        fused_expert_params_mapping = [
            (f"experts.{base_layer}w13_weight", "experts.gate_up_proj", 0, "w1"),
            (f"experts.{base_layer}w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # qwen3.5 no need to transpose
                        # loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # Skip loading extra bias for GPTQ models.
                        if (
                            name_mapped.endswith(".bias")
                            or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################

from collections.abc import Sequence
from typing import Any

from vllm.inputs.engine import mm_input
from vllm.multimodal.inputs import PlaceholderRange
from vllm.renderers.inputs.preprocess import parse_model_prompt


class Qwen3_5InputsProcessor:
    """Flat multimodal inputs processor for Qwen3.5-VL.

    Zero inheritance, zero abstract deps.
    Directly calls HF ``Qwen3VLProcessor`` and builds ``MultiModalInput``.

    Usage::

        proc = Qwen3_5InputsProcessor(model_config)
        result: MultiModalInput = proc(prompt_token_ids, mm_data)
    """

    def __init__(self, model_config):
        from vllm.tokenizers.registry import cached_tokenizer_from_config
        from vllm.transformers_utils.processor import cached_processor_from_config

        self.model_config = model_config
        self.tokenizer = cached_tokenizer_from_config(model_config)
        self.hf_processor = cached_processor_from_config(model_config)

        hf_config = model_config.hf_config
        vision_cfg = hf_config.vision_config
        self._spatial_merge_size = vision_cfg.spatial_merge_size
        self._vision_start_id = hf_config.vision_start_token_id
        self._vision_end_id = hf_config.vision_end_token_id
        self._video_token_id = hf_config.video_token_id
        self._image_token_id = self.hf_processor.image_token_id
        self._temporal_patch_size = (
            self.hf_processor.video_processor.temporal_patch_size
        )

    def apply(
        self,
        prompt_token_ids: list[int],
        mm_data: dict[str, object],
    ) -> dict:
        """Process multimodal data and return ``MultiModalInput``.

        Args:
            prompt_token_ids: Token IDs with ``<|image_pad|>`` /
                ``<|vision_start|><|video_pad|><|vision_end|>`` placeholders.
            mm_data: Raw multimodal data, e.g.
                ``{"image": [PIL.Image, ...], "video": [(array, meta), ...]}``.
        """
        images = mm_data.get("image") or []
        if not isinstance(images, list):
            images = [images]
        videos = mm_data.get("video") or []
        if not isinstance(videos, list):
            videos = [videos]

        # --- Process images ---
        image_results: list[dict] = []
        if images:
            img_out = self.hf_processor(
                text="",
                images=list(images),
                return_tensors="pt",
            )
            grid_thw_all = img_out["image_grid_thw"]
            for i in range(len(images)):
                grid_thw = grid_thw_all[i]  # shape (3,)
                num_tokens = int(grid_thw.prod()) // (self._spatial_merge_size**2)
                image_results.append(
                    {
                        "pixel_values": img_out["pixel_values"],
                        "image_grid_thw": grid_thw,
                        "num_tokens": num_tokens,
                        "repl_tokens": [self._image_token_id] * num_tokens,
                    }
                )

        # --- Process videos ---
        video_results: list[dict] = []
        for video_item in videos:
            video_array, metadata = video_item
            vid_out = self.hf_processor(
                text="<|vision_start|><|video_pad|><|vision_end|>",
                videos=[[video_array]],
                video_metadata=[[metadata]],
                return_tensors="pt",
            )

            grid_thw = vid_out["video_grid_thw"]
            num_frames = int(grid_thw[0, 0])
            tokens_per_frame_base = int(grid_thw[0, 1:].prod()) // (
                self._spatial_merge_size**2
            )
            tokens_per_frame = [tokens_per_frame_base] * num_frames

            timestamps = self._compute_timestamps(metadata)

            repl_tokens = self._build_video_tokens(tokens_per_frame, timestamps)

            video_results.append(
                {
                    "pixel_values_videos": vid_out["pixel_values_videos"],
                    "video_grid_thw": grid_thw,
                    "timestamps": timestamps,
                    "repl_tokens": repl_tokens,
                }
            )

        # --- Apply prompt updates ---
        new_ids, mm_kwargs, placeholders = self._apply_prompt_updates(
            list(prompt_token_ids), image_results, video_results
        )

        # --- Build MultiModalInput ---
        mm_hashes: dict[str, list[str]] = {}
        if image_results:
            mm_hashes["image"] = [f"img-{i}" for i in range(len(image_results))]
        if video_results:
            mm_hashes["video"] = [f"vid-{i}" for i in range(len(video_results))]

        return mm_input(new_ids, mm_kwargs, mm_hashes, placeholders)

    def preprocess_cmpl(
        self,
        prompts: Sequence[Any],
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Full preprocessing: raw prompt → EngineInput list.

        Takes the same prompt format as ``LLM.generate()`` (str, TokensPrompt,
        etc.) and returns a list of ``EngineInput`` ready for ``AsyncLLM``.
        """
        model_config = self.model_config
        results: list[dict] = []
        tok_kwargs = tokenization_kwargs or {}

        for prompt in prompts:
            parsed = parse_model_prompt(model_config, prompt)

            # Tokenize
            prompt_text = parsed.get("prompt", parsed.get("prompt_token_ids"))
            if isinstance(prompt_text, str):
                prompt_token_ids = self.tokenizer.encode(prompt_text, **tok_kwargs)
            else:
                prompt_token_ids = prompt_text

            mm_data = parsed.get("multi_modal_data")

            if mm_data:
                engine_input = self.apply(prompt_token_ids, mm_data)
            else:
                engine_input = {"type": "token", "prompt_token_ids": prompt_token_ids}

            if cache_salt := parsed.get("cache_salt"):
                engine_input["cache_salt"] = cache_salt

            results.append(engine_input)

        return results

    # ------------------------------------------------------------------
    # Prompt update: scan & replace placeholder tokens
    # ------------------------------------------------------------------

    def _apply_prompt_updates(
        self,
        ids: list[int],
        image_results: list[dict],
        video_results: list[dict],
    ) -> tuple[list[int], dict, dict]:
        placeholders: dict[str, list] = {"image": [], "video": []}
        mm_kwargs: dict[str, list] = {"image": [], "video": []}

        # Images: replace contiguous <|image_pad|> blocks
        img_idx = 0
        i = 0
        while i < len(ids):
            if ids[i] == self._image_token_id and img_idx < len(image_results):
                start = i
                while i < len(ids) and ids[i] == self._image_token_id:
                    i += 1
                result = image_results[img_idx]
                repl = result["repl_tokens"]
                ids[start:i] = repl
                placeholders["image"].append(
                    PlaceholderRange(offset=start, length=len(repl))
                )
                mm_kwargs["image"].append(self._build_image_kwarg(result, img_idx))
                img_idx += 1
            else:
                i += 1

        # Videos: replace <|vision_start|><|video_pad|><|vision_end|>
        video_target = [
            self._vision_start_id,
            self._video_token_id,
            self._vision_end_id,
        ]
        vid_idx = 0
        i = 0
        while i < len(ids):
            if ids[i : i + 3] == video_target and vid_idx < len(video_results):
                result = video_results[vid_idx]
                repl = result["repl_tokens"]
                ids[i : i + 3] = repl
                placeholders["video"].append(
                    PlaceholderRange(offset=i, length=len(repl))
                )
                mm_kwargs["video"].append(self._build_video_kwarg(result))
                vid_idx += 1
                i += len(repl)
            else:
                i += 1

        if not mm_kwargs["image"]:
            del mm_kwargs["image"]
            del placeholders["image"]
        if not mm_kwargs["video"]:
            del mm_kwargs["video"]
            del placeholders["video"]

        return ids, mm_kwargs, placeholders

    # ------------------------------------------------------------------
    # Build mm_kwargs items
    # ------------------------------------------------------------------

    def _build_image_kwarg(self, result: dict, idx: int) -> dict:
        """Build a single image kwarg dict (modality-level tensors)."""
        from vllm.multimodal.inputs import (
            MultiModalBatchedField,
            MultiModalFieldElem,
            MultiModalKwargsItem,
            MultiModalSharedField,
        )

        grid_thw = result["image_grid_thw"]
        pixel_grid_size = int(grid_thw.prod())

        return MultiModalKwargsItem(
            {
                "pixel_values": MultiModalFieldElem(
                    data=result["pixel_values"],
                    field=MultiModalSharedField(batch_size=pixel_grid_size),
                ),
                "image_grid_thw": MultiModalFieldElem(
                    data=grid_thw,  # shape (3,) → reduce_data stacks to (M, 3)
                    field=MultiModalBatchedField(keep_on_cpu=True),
                ),
            }
        )

    def _build_video_kwarg(self, result: dict) -> dict:
        """Build a single video kwarg dict."""
        import torch

        from vllm.multimodal.inputs import (
            MultiModalBatchedField,
            MultiModalFieldElem,
            MultiModalKwargsItem,
            MultiModalSharedField,
        )

        grid_thw = result["video_grid_thw"]
        video_grid_size = int(grid_thw.prod())
        timestamps = result["timestamps"]
        if not isinstance(timestamps, torch.Tensor):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)

        return MultiModalKwargsItem(
            {
                "pixel_values_videos": MultiModalFieldElem(
                    data=result["pixel_values_videos"],
                    field=MultiModalSharedField(batch_size=video_grid_size),
                ),
                "video_grid_thw": MultiModalFieldElem(
                    data=grid_thw,  # shape (1, 3) → reduce_data stacks to (M, 3)
                    field=MultiModalBatchedField(keep_on_cpu=True),
                ),
                "timestamps": MultiModalFieldElem(
                    data=timestamps,  # shape (num_frames,) → reduce_data stacks
                    field=MultiModalBatchedField(keep_on_cpu=True),
                ),
            }
        )

    # ------------------------------------------------------------------
    # Video helpers (inlined from Qwen3VLMultiModalProcessor)
    # ------------------------------------------------------------------

    def _compute_timestamps(self, metadata: dict) -> list[float]:
        """Compute per-frame timestamps in seconds."""
        merge_size = self._spatial_merge_size
        indices = metadata["frames_indices"]
        video_fps = metadata["fps"]

        do_sample = metadata.get("do_sample_frames", False)
        if do_sample:
            total = metadata["total_num_frames"]
            vp = self.hf_processor.video_processor
            num_frames = int(total / metadata["fps"] * vp.fps)
            num_frames = min(min(max(num_frames, vp.min_frames), vp.max_frames), total)
            indices = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()

        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
        raw_ts = [idx / video_fps for idx in indices]
        return [
            (raw_ts[i] + raw_ts[i + merge_size - 1]) / 2
            for i in range(0, len(raw_ts), merge_size)
        ]

    def _build_video_tokens(
        self,
        tokens_per_frame: list[int],
        timestamps: list[float],
    ) -> list[int]:
        """Build per-frame token sequence for a video.

        Structure per frame::

            [timestamp_tokens] + [vision_start] + [video_token] * N + [vision_end]
        """
        assert len(timestamps) == len(tokens_per_frame)
        all_ids: list[int] = []
        for ts, n_tok in zip(timestamps, tokens_per_frame):
            ts_text = f"<{ts:.1f} seconds>"
            ts_ids = self.tokenizer.encode(ts_text, add_special_tokens=False)
            all_ids.extend(ts_ids)
            all_ids.append(self._vision_start_id)
            all_ids.extend([self._video_token_id] * n_tok)
            all_ids.append(self._vision_end_id)
        return all_ids


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # Qwen3.5 does not support multimodal pruning (EVS).
    supports_multimodal_pruning = False

    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3.5.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have not __init__ method, so we need to use nn.Module.__init__
        nn.Module.__init__(self)
        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        # Qwen3.5 does not support multimodal pruning (EVS).
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
