# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Image processor for Unlimited-OCR (baidu/Unlimited-OCR)."""

import math

from PIL import Image

from vllm.logger import init_logger
from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor

logger = init_logger(__name__)

# Page delimiter of the Unlimited-OCR-2 multi-page layout; one page marker per
# page precedes its ``<image>`` placeholder in both prompt and output.
PAGE_TOKEN = "<PAGE>"

# Trailing instruction of the trained multi-page layout
# (``"<PAGE><image>" * n_pages + MULTI_PAGE_INSTRUCTION``).
MULTI_PAGE_INSTRUCTION = "Multi page parsing."


def build_multi_page_prompt(
    conversation: str,
    num_images: int,
    image_token: str = "<image>",
) -> str:
    """Expand a multi-page prompt to the ``<PAGE><image>`` per-page layout.

    Mirrors the reference ``infer`` prompt construction: a prompt without
    ``<image>`` placeholders gets ``<PAGE><image>`` per page prepended, and a
    single ``<image>`` placeholder with several pages is expanded to one
    ``<PAGE><image>`` pair per page. A prompt that already carries one
    placeholder per page is returned unchanged.

    Args:
        conversation: The prompt text.
        num_images: Number of page images in the request.
        image_token: The image placeholder string.

    Returns:
        The expanded prompt.

    Raises:
        ValueError: If the placeholder count matches neither 0, 1 nor the
            number of images.

    """
    num_placeholders = conversation.count(image_token)
    if num_placeholders == num_images:
        return conversation
    page_prefix = f"{PAGE_TOKEN}{image_token}" * num_images
    if num_placeholders == 0:
        return page_prefix + conversation
    if num_placeholders == 1:
        return conversation.replace(image_token, page_prefix)
    raise ValueError(
        f"Prompt has {num_placeholders} {image_token!r} placeholders but "
        f"{num_images} images."
    )


class UnlimitedOCRProcessor(DeepseekOCRProcessor):
    """DeepseekOCRProcessor variant for Unlimited-OCR.

    Behavioural differences from the base processor:

    * Multi-image safeguard: when more than one image is present, crop
      ("gundam") mode is disabled, so every page is encoded in base mode.
    * Multi-page prompt layout (Unlimited-OCR-2): multi-image prompts are
      expanded to ``"<PAGE><image>" * n_pages + instruction`` via
      :func:`build_multi_page_prompt`, matching the layout the SR-SWA model
      was trained with.

    Because the effective crop flag then depends on *how many* images are in the
    request, the per-item processing output is no longer invariant of sibling
    images. ``UnlimitedOCRMultiModalProcessor`` accounts for this by bypassing
    the multimodal processing cache for multi-image requests (see its
    ``_cached_apply_hf_processor``), so the two paths stay consistent.

    DeepSeek-OCR does *not* have this restriction because its ``max_crops=6``
    is small enough to be safe for multi-image use.
    """

    @property
    def page_token_id(self) -> int | None:
        """Vocab id of ``<PAGE>``; None for v1 tokenizers without it."""
        return self.tokenizer.vocab.get(PAGE_TOKEN)

    def get_num_base_mode_image_tokens(self) -> int:
        """Image tokens per page in base (non-crop) mode.

        For the default gundam stack (``base_size=1024``, ``strategy="v1"``)
        this is 273, matching ``sparse_attention_config.per_page_image_tokens``
        of Unlimited-OCR-2.
        """
        num_queries_base = math.ceil(
            (self.base_size // self.patch_size) / self.downsample_ratio
        )
        num_tokens_base = (
            num_queries_base * (num_queries_base + 1)
            if self.strategy == "v1"
            else num_queries_base * num_queries_base
        )
        # Plus the single newline image token appended after the global view.
        return num_tokens_base + 1

    def tokenize_with_images(
        self,
        conversation: str,
        images: list[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        if len(images) > 1:
            if cropping:
                logger.warning_once(
                    "Unlimited-OCR: crop mode is not supported for multi-image "
                    "input. Falling back to cropping=False."
                )
                cropping = False
            conversation = build_multi_page_prompt(
                conversation, len(images), self.image_token
            )
        return super().tokenize_with_images(
            conversation, images, bos=bos, eos=eos, cropping=cropping
        )
