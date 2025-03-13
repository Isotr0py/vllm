# SPDX-License-Identifier: Apache-2.0

# adapted from https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B/blob/main/modeling_internvl_chat_hico2.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import (Callable, List, Literal, Optional, Set, Tuple, TypedDict, TypeVar,
                    Union)

import torch
import numpy.typing as npt
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .internvl import (IMG_CONTEXT, IMG_END, IMG_START,
                       BaseInternVLProcessingInfo, BaseInternVLProcessor,
                       InternVLProcessor, InternVLProcessingInfo,
                       InternVLChatModel, InternVLDummyInputsBuilder,
                       InternVLMultiModalProcessor, build_transform,
                       find_closest_aspect_ratio, get_internvl_target_ratios, image_to_pixel_values_internvl)


class InternVLHiCoProcessor:
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        image_size: int = config.vision_config.image_size

        self.local_num_frames = 4
        self.num_tome_tokens = 64
        self.num_image_token = self.num_tome_tokens // self.local_num_frames
        self.image_size = image_size
        self.use_thumbnail: bool = config.use_thumbnail

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

    def get_video_repl(
        self,
        feature_size: int,
        num_frames: int,
    ) -> str:
        raise NotImplementedError

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

        return get_internvl_target_ratios(min_num, max_num)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_internvl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
    ) -> list[torch.Tensor]:
        videos_pil = [[
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode="RGB")
            for frame in frames.astype(np.uint8)
        ] for frames in videos]
        return [
            torch.cat([
                image_to_pixel_values_internvl(
                    frame,
                    input_size=self.image_size,
                    min_num=1,
                    max_num=1,
                    use_thumbnail=False,
                ) for frame in frames
            ]) for frames in videos_pil
        ]

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        videos: Optional[Union[npt.NDArray, list[npt.NDArray]]] = None,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]

        if videos is None:
            videos = []
        elif not isinstance(videos, list):
            videos = [videos]

        if len(videos) == 0:
            video_inputs = {}
        else:
            pixel_values_lst = self._videos_to_pixel_values_lst(videos, )
            video_inputs = {
                "pixel_values_flat_video": torch.cat(pixel_values_lst),
                "video_num_patches": list(map(len, pixel_values_lst)),
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token

                # image_repl = self.get_video_repl(feature_size, num_patches)
                # text = [t.replace('<video>', image_repl, 1) for t in text]

        text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **video_inputs,
            },
            tensor_type=return_tensors,
        )



# copied from https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B/blob/main/modeling_internvl_chat_hico2.py
def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c),
                              src)  # , reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n,
                          metric.shape[1],
                          c,
                          device=x.device,
                          dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2,
                     index=(2 * unm_idx).expand(n, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


# copied from https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B/blob/main/modeling_internvl_chat_hico2.py
def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


class InternVLHiCoProcessingInfo(InternVLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        **kwargs: object,
    ) -> InternVLProcessor:
        processor = super().get_hf_processor(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            **kwargs,
        )
        local_num_frames = 4
        num_tome_tokens = 64
        processor.num_image_token = num_tome_tokens // local_num_frames
        return processor


@MULTIMODAL_REGISTRY.register_processor(
    InternVLMultiModalProcessor,
    info=InternVLHiCoProcessingInfo,
    dummy_inputs=InternVLDummyInputsBuilder)
class InternVLChatHiCoModel(InternVLChatModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.local_num_frames = 4
        self.num_tome_tokens = 64
        self.num_image_token = self.num_tome_tokens // self.local_num_frames

    def merge_tokens(self, x: torch.Tensor, target_num_token: int):
        r"""
        x = torch.randn(10, 2560, c)
        x = merge_tokens(x, r_merge_list=[1280])
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        assert tmp_p > target_num_token, (
            f"{tmp_p} should greater than {target_num_token}")
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)

        head = self.config.text_config.num_attention_heads

        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2)  # [b, p, c//head]
            merge, _ = bipartite_soft_matching(metric, r)
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        # x = x.reshape(-1, c)  # 300, 1024
        return x

    def extract_feature(self, pixel_values: torch.Tensor):
        vit_embeds = self.vision_model(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        assert vit_embeds.shape[0] % self.local_num_frames == 0
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0] // self.local_num_frames, -1,
            vit_embeds.shape[-1])
        vit_embeds = self.merge_tokens(vit_embeds, self.num_tome_tokens)
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0] * self.local_num_frames, -1,
            vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds