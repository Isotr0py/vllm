# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.transformers_utils.processors.deepseek_ocr import calculate_aspect_ratios
from vllm.transformers_utils.processors.unlimited_ocr import (
    build_multi_page_prompt,
)

from ...utils import build_model_context


@pytest.mark.usefixtures("default_vllm_config")
def test_profiling_tokens_cover_all_crop_grids():
    """Every supported crop grid fits the profiled image token budget."""
    ctx = build_model_context(
        "baidu/Unlimited-OCR",
        limit_mm_per_prompt={"image": 1},
        model_config_kwargs={"max_model_len": 8192},
    )
    mm_processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = mm_processor.info.get_hf_processor()
    dummy_inputs = mm_processor.get_dummy_mm_inputs({"image": 1})
    budget = dummy_inputs["mm_placeholders"]["image"][0].get_num_embeds()

    for width_tiles, height_tiles in calculate_aspect_ratios(
        max_num=hf_processor.max_crops
    ):
        num_tokens = mm_processor.info.get_num_image_tokens(
            image_width=width_tiles * hf_processor.image_size,
            image_height=height_tiles * hf_processor.image_size,
        )
        assert num_tokens <= budget, (width_tiles, height_tiles, num_tokens, budget)


@pytest.mark.parametrize("use_uuid", [False, True])
@pytest.mark.parametrize("use_embeds", [False, True])
def test_image_cache_identity_across_image_counts(use_uuid: bool, use_embeds: bool):
    config = ModelConfig(
        model="baidu/Unlimited-OCR",
        limit_mm_per_prompt={"image": 3},
        enable_mm_embeds=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    tokenizer = processor.info.get_tokenizer()
    if use_embeds:
        images = [torch.ones(8, config.get_hidden_size())] * 3
    else:
        images = [
            Image.new("RGB", (1280, 800), color) for color in ("red", "green", "blue")
        ]
    hashes, lengths = [], []
    for count in (1, 2, 3):
        inputs = ProcessorInputs(
            prompt=tokenizer.encode("<image>\n" * count + "Describe."),
            mm_data_items=processor.info.parse_mm_data({"image": images[:count]}),
            mm_uuid_items={"image": [f"image-{i}" for i in range(count)]}
            if use_uuid
            else None,
        )
        if use_embeds:
            hashes.append(processor._get_mm_hashes(inputs)["image"][0])
            continue
        result = processor.apply(inputs, TimingContext(enabled=False))
        hashes.append(result["mm_hashes"]["image"][0])
        lengths.append(result["mm_placeholders"]["image"][0].length)
    assert hashes[1] == hashes[2]
    if use_embeds:
        assert hashes[0] == hashes[1]
        if use_uuid:
            assert hashes[0] == "image-0"
    else:
        assert lengths[0] > lengths[1] == lengths[2]
        assert hashes[0] != hashes[1]


@pytest.mark.parametrize("kwargs_on_init", [False, True])
def test_image_cache_identity_without_cropping(kwargs_on_init: bool):
    mm_processor_kwargs = {"crop_mode": False}
    config = ModelConfig(
        model="baidu/Unlimited-OCR",
        limit_mm_per_prompt={"image": 2},
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    images = [Image.new("RGB", (1280, 800), color) for color in ("red", "blue")]
    cache = MultiModalProcessorOnlyCache(config)
    results = []
    for count in (1, 2, 1):
        inputs = ProcessorInputs(
            prompt=processor.info.get_tokenizer().encode("<image>\n" * count),
            mm_data_items=processor.info.parse_mm_data({"image": images[:count]}),
            hf_processor_mm_kwargs={} if kwargs_on_init else mm_processor_kwargs,
            cache=cache,
        )
        results.append(processor.apply(inputs, TimingContext(enabled=False)))
    assert results[0]["mm_hashes"]["image"][0] == results[1]["mm_hashes"]["image"][0]
    assert (
        results[0]["mm_placeholders"]["image"][0].length
        == results[1]["mm_placeholders"]["image"][0].length
    )
    assert results[0]["prompt_token_ids"] == results[2]["prompt_token_ids"]


def test_build_multi_page_prompt():
    """String-level <PAGE> layout expansion rules of the reference infer()."""
    assert (
        build_multi_page_prompt("Multi page parsing.", 3)
        == "<PAGE><image>" * 3 + "Multi page parsing."
    )
    # A single placeholder with several pages expands to one pair per page.
    assert build_multi_page_prompt("<image>Parse.", 2) == "<PAGE><image>" * 2 + "Parse."
    # Already-expanded prompts pass through unchanged.
    already = "<PAGE><image>" * 2 + "Multi page parsing."
    assert build_multi_page_prompt(already, 2) == already
    # A placeholder count matching neither 0, 1 nor the image count fails.
    with pytest.raises(ValueError, match="placeholders"):
        build_multi_page_prompt("<image><image><image>", 2)


def test_v1_config_has_no_sparse_attention_config():
    """v1 configs must keep parsing with sparse_attention_config unset."""
    ctx = build_model_context(
        "baidu/Unlimited-OCR",
        limit_mm_per_prompt={"image": 1},
        model_config_kwargs={"max_model_len": 8192},
    )
    mm_processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert mm_processor.info.get_hf_config().sparse_attention_config is None


def test_multi_page_prompt_expansion_and_token_counts():
    """Multi-page requests expand to '<PAGE><image>' * n + instruction, with
    one base-mode 273-token view per page (no local crops)."""
    config = ModelConfig(
        model="baidu/Unlimited-OCR-2",
        limit_mm_per_prompt={"image": 3},
        max_model_len=8192,
    )
    mm_processor = MULTIMODAL_REGISTRY.create_processor(config)

    sparse_config = mm_processor.info.get_hf_config().sparse_attention_config
    assert sparse_config is not None
    assert sparse_config.page_token_id == 128827
    assert sparse_config.per_page_image_tokens == 273

    hf_processor = mm_processor.info.get_hf_processor()
    assert hf_processor.page_token_id == sparse_config.page_token_id
    assert (
        hf_processor.get_num_base_mode_image_tokens()
        == sparse_config.per_page_image_tokens
    )

    pages = [Image.new("RGB", (1280, 800), color) for color in ("red", "green", "blue")]
    variants = (
        "Multi page parsing.",
        "<image>Multi page parsing.",
        "<PAGE><image>" * 3 + "Multi page parsing.",
    )
    for prompt in variants:
        (
            input_ids,
            _,
            images_crop,
            _,
            images_spatial_crop,
            num_image_tokens,
            _,
        ) = hf_processor.tokenize_with_images(prompt, pages)
        assert num_image_tokens == [273] * 3
        assert input_ids[0].tolist().count(hf_processor.page_token_id) == 3
        # Multi-page requests always fall back to base mode: a single global
        # view per page and no local crops.
        assert images_crop.shape[0] == 0
        assert images_spatial_crop.tolist() == [[1, 1]] * 3


def test_multi_page_apply_produces_page_markers_and_273_token_pages():
    """The vLLM processing pipeline keeps one <PAGE> id per page in the final
    prompt and expands every <image> placeholder to 273 image tokens."""
    config = ModelConfig(
        model="baidu/Unlimited-OCR-2",
        limit_mm_per_prompt={"image": 3},
        max_model_len=8192,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    tokenizer = processor.info.get_tokenizer()
    page_token_id = tokenizer.vocab["<PAGE>"]

    images = [
        Image.new("RGB", (1280, 800), color) for color in ("red", "green", "blue")
    ]
    prompt = "<PAGE><image>" * 3 + "Multi page parsing."
    inputs = ProcessorInputs(
        prompt=tokenizer.encode(prompt),
        mm_data_items=processor.info.parse_mm_data({"image": images}),
    )
    result = processor.apply(inputs, TimingContext(enabled=False))

    assert result["prompt_token_ids"].count(page_token_id) == 3
    placeholders = result["mm_placeholders"]["image"]
    assert [p.length for p in placeholders] == [273] * 3
