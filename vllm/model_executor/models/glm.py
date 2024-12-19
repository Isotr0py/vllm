"""Inference-only HF format GLM-4 model compatible with THUDM weights."""
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig, QuantizationConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.glm4_vision_encoder import GLU
from vllm.model_executor.models.siglip import SiglipEncoder

from .utils import PPMissingLayer


class Adapter(nn.Module):

    def __init__(
        self,
        vision_hidden_size: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.boi = nn.Parameter(torch.ones(1, 1, hidden_size).float())
        self.eoi = nn.Parameter(torch.ones(1, 1, hidden_size).float())
        self.conv = nn.Conv2d(in_channels=vision_hidden_size,
                              out_channels=hidden_size,
                              kernel_size=2,
                              stride=2)
        self.linear_proj = GLU(hidden_size=hidden_size,
                               ffn_hidden_size=intermediate_size,
                               quant_config=quant_config)

    def forward(self, image_emb):
        b, s, e = image_emb.shape  # (b, 6400, 1792)
        grid_size = int(s**0.5)
        image_emb = image_emb.view(b, grid_size, grid_size,
                                   e).permute(0, 3, 1, 2)  # (b, 1792, 80, 80)
        image_emb = self.conv(image_emb)  # (b, 4096, 40, 40)
        image_emb = image_emb.flatten(2).transpose(1, 2)  # (b, 1600, 4096)
        image_emb = self.linear_proj(image_emb)  # (b, 1600, 6656)
        image_emb = torch.cat([
            self.boi.repeat(len(image_emb), 1, 1), image_emb,
            self.eoi.repeat(len(image_emb), 1, 1)
        ],
                              dim=1)
        return image_emb


class VisionModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix="",
    ):
        super().__init__()
        vision_config = config.vision_config
        self.vit = SiglipEncoder(
            config.vision_config,
            quant_config=quant_config,
            prefix=f"{prefix}.vit",
        )
        self.adapter = Adapter(
            vision_hidden_size=vision_config.hidden_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

    def forward(self, pixel_values: torch.Tensor):
        vit_output = self.vit(pixel_values)
        return self.adapter(vit_output)


class GlmForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Hack Llama model to fit HF format GLM implementation
        # Attention difference between GLM and Llama:
        # 1. Half partial rotary_dim and no Neox style.
        # 2. There is no bias for o_proj in attention
        for layer in self.model.layers:
            if not isinstance(layer, PPMissingLayer):
                layer.self_attn.rotary_emb.rotary_dim //= 2
                layer.self_attn.rotary_emb.is_neox_style = False
                layer.self_attn.o_proj.bias = None
                layer.self_attn.o_proj.skip_bias_add = True

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        # Initialize VL
        if hasattr(config, "vision_config"):
            self.vision = VisionModel(
                config=config,
                quant_config=quant_config,
                prefix="vision",
            )
