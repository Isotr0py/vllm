# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for minimal media-free multimodal KV continuation.

A decode instance without media or encoder capacity must be able to continue
a multimodal request purely from existing decoder KV: admission accepts
hash-only features with `requires_kv`, the media cache skips them, and the
scheduler fails the request (rather than computing) whenever coverage is
insufficient.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    engine_input_from_features,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalInput, TokensPrompt, mm_input
from vllm.multimodal.cache import (
    LruKeyReplicatedReceiverCache,
    MultiModalCacheMissError,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.renderers.base import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus


def kv_features() -> MultiModalFeatures:
    return MultiModalFeatures(
        mm_hashes={"image": ["image-hash"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=1, length=3)]},
        requires_kv=True,
    )


def kv_engine_input() -> MultiModalInput:
    return engine_input_from_features([1, 2, 3, 4, 5], kv_features())


def admission_processor() -> InputProcessor:
    processor = object.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(max_model_len=64, runner_type="generate")
    processor.skip_prompt_length_check = False
    processor.supports_mm_inputs = True
    # A decode worker started with --limit-mm-per-prompt '{"image": 0}'.
    processor.mm_encoder_cache_size = 0
    processor.renderer = SimpleNamespace(validate_token_ids=Mock())
    return processor


def test_requires_kv_omitted_from_wire_format_when_false():
    features = MultiModalFeatures(
        mm_hashes={"image": ["image-hash"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=1, length=3)]},
    )
    assert "requires_kv" not in json.dumps(features.model_dump(mode="json"))
    assert '"requires_kv":true' in json.dumps(
        kv_features().model_dump(mode="json")
    ).replace(" ", "")


def test_requires_kv_rejects_media_payload_and_empty_hash():
    with pytest.raises(ValueError, match="media payload"):
        MultiModalFeatures(
            mm_hashes={"image": ["image-hash"]},
            mm_placeholders={"image": [PlaceholderRangeInfo(offset=1, length=3)]},
            kwargs_data={"image": ["deadbeef"]},
            requires_kv=True,
        )
    with pytest.raises(ValueError, match="nonempty hash"):
        MultiModalFeatures(
            mm_hashes={"image": [""]},
            mm_placeholders={"image": [PlaceholderRangeInfo(offset=1, length=3)]},
            requires_kv=True,
        )


def test_kv_only_input_admitted_with_zero_encoder_budget():
    engine_input = kv_engine_input()
    assert engine_input["mm_requires_kv"] is True
    assert engine_input["mm_kwargs"] == {"image": [None]}
    admission_processor()._validate_model_inputs(
        None, engine_input, SamplingParams(max_tokens=1)
    )

    # Control: the same prompt carrying media is rejected at zero budget.
    media_input = mm_input(
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_kwargs={"image": [MultiModalKwargsItem.dummy()]},
        mm_hashes={"image": ["image-hash"]},
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=3)]},
    )
    with pytest.raises(VLLMValidationError, match="encoder cache size"):
        admission_processor()._validate_model_inputs(
            None, media_input, SamplingParams(max_tokens=1)
        )


def test_kv_only_input_with_media_payload_rejected():
    engine_input = kv_engine_input()
    engine_input["mm_kwargs"] = MultiModalKwargsItems[MultiModalKwargsItem | None](
        {"image": [MultiModalKwargsItem.dummy()]}
    )
    with pytest.raises(VLLMValidationError, match="media payload"):
        admission_processor()._validate_model_inputs(
            None, engine_input, SamplingParams(max_tokens=1)
        )


def test_kv_only_input_rejects_prompt_scoring():
    processor = admission_processor()
    with pytest.raises(VLLMValidationError, match="prompt_logprobs"):
        processor._validate_model_inputs(
            None, kv_engine_input(), SamplingParams(max_tokens=1, prompt_logprobs=5)
        )
    with pytest.raises(VLLMValidationError, match="prompt_logprobs"):
        processor._validate_model_inputs(
            None,
            kv_engine_input(),
            SamplingParams(max_tokens=1, prompt_logprob_token_ids=[[1]]),
        )


def test_receiver_cache_skips_kv_only_features():
    cache = LruKeyReplicatedReceiverCache(
        SimpleNamespace(
            get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=0.01)
        )
    )
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="image-hash",
        mm_position=PlaceholderRange(offset=1, length=3),
        mm_hash="image-hash",
        requires_kv=True,
    )
    assert cache.get_and_update_features([feature])[0].data is None
    feature.requires_kv = False
    with pytest.raises(MultiModalCacheMissError):
        cache.get_and_update_features([feature])


def test_renderer_rejects_rerendering_kv_only_input():
    with pytest.raises(VLLMValidationError, match="cannot be rendered again"):
        BaseRenderer._reject_kv_only_prompt(kv_engine_input())
    BaseRenderer._reject_kv_only_prompt(TokensPrompt(prompt_token_ids=[1, 2]))


def test_request_caches_required_kv_boundary():
    feature = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="image-hash",
        mm_position=PlaceholderRange(offset=50, length=100),
        requires_kv=True,
    )
    request = Request(
        request_id="kv",
        prompt_token_ids=[0] * 160,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=[feature],
    )
    assert request.mm_required_kv_tokens == 150
    request.mm_features[0].requires_kv = False
    plain = Request(
        request_id="plain",
        prompt_token_ids=[0] * 160,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=[feature],
    )
    assert plain.mm_required_kv_tokens == 0


def test_scheduler_fails_request_without_full_kv_coverage():
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    request = create_requests(
        num_requests=1,
        num_tokens=160,
        mm_positions=[[PlaceholderRange(offset=50, length=100)]],
        mm_requires_kv=True,
    )[0]
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert request.request_id not in output.num_scheduled_tokens
    assert not output.scheduled_encoder_inputs
    assert request.request_id in scheduler.missing_multimodal_kv_reqs

    result = scheduler.update_from_output(
        output, ModelRunnerOutput(req_ids=[], req_id_to_index={})
    )
    assert request.status == RequestStatus.FINISHED_ERROR
    assert result[request.client_index].outputs[0].finish_reason == FinishReason.ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.missing_multimodal_kv_reqs


def test_scheduler_skips_encoder_work_when_kv_covers_media():
    scheduler = create_scheduler(model="llava-hf/llava-1.5-7b-hf")
    request = create_requests(
        num_requests=1,
        num_tokens=160,
        mm_positions=[[PlaceholderRange(offset=50, length=100)]],
        mm_requires_kv=True,
    )[0]
    # KV already covers the image span [50, 150); only text continues here.
    assert scheduler._try_schedule_encoder_inputs(request, 150, 10, 100) == (
        [],
        10,
        100,
        [],
        [],
    )
    assert not scheduler.missing_multimodal_kv_reqs

    # Preemption drops the computed prefix; coverage is re-checked on resume
    # and the request must fail rather than recompute the missing media.
    request.status = RequestStatus.RUNNING
    scheduler._preempt_request(request, 0)
    assert request.num_computed_tokens == 0
    assert scheduler._try_schedule_encoder_inputs(request, 0, 160, 100) == (
        [],
        0,
        100,
        [],
        [],
    )
    assert request.request_id in scheduler.missing_multimodal_kv_reqs
