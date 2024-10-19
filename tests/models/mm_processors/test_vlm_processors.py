from typing import Union, List, Optional

import pytest
import torch
from transformers import AutoProcessor, AutoImageProcessor, AutoTokenizer
from PIL.Image import Image

from vllm.inputs import InputRegistry, InputContext, LLMInputs
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalInputs,
                             MultiModalRegistry)

from ...models.utils import build_model_context


def compare_inputs_processor(model, prompt: str, images: Optional[Union[Image, List[Image]]]) -> None:
    input_registry = InputRegistry()
    ctx: InputContext = build_model_context(
        tokenizer_name=model,
        trust_remote_code=True,
    )

    processor = input_registry.create_input_processor(ctx.model_config)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)

    llm_inputs = LLMInputs(prompt_token_ids=tokenizer.encode(prompt),
                           prompt=prompt,
                           multi_modal_data={"image": images})

    vllm_proc_inputs = processor(
        ctx=ctx,
        llm_inputs=llm_inputs,
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", [torch.half])
def test_vlm_processors(dist_init, image_assets, model, dtype: str) -> None:
    run_intern_vit_test(
        image_assets,
        model,
        dtype=dtype,
    )