# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Unlimited-OCR-2 PageCountGuard logits processor."""

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.unlimited_ocr_page_guard import (
    PageCountGuardLogitsProcessor,
    PageCountGuardPerReqLogitsProcessor,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate

PAGE_ID = 128827
EOS_ID = 1
VOCAB_SIZE = 129280


def _logits() -> torch.Tensor:
    return torch.zeros(VOCAB_SIZE)


def _step(
    guard: PageCountGuardLogitsProcessor,
    prompt_ids: list[int],
    output_ids: list[int],
) -> torch.Tensor:
    return guard(prompt_ids, output_ids, _logits())


def test_eos_masked_until_all_pages_done():
    prompt = [PAGE_ID] * 2 + [100]
    guard = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID)
    output_ids = [10, 11]
    logits = _step(guard, prompt, output_ids)
    assert logits[EOS_ID].item() == -float("inf")
    # Only EOS is masked; everything else is untouched.
    assert torch.isfinite(logits).sum().item() == VOCAB_SIZE - 1

    output_ids += [PAGE_ID]
    logits = _step(guard, prompt, output_ids)
    assert logits[EOS_ID].item() == -float("inf")

    output_ids += [12, PAGE_ID]
    logits = _step(guard, prompt, output_ids)
    assert torch.isfinite(logits).all()


def test_page_cap_forces_page_token_or_eos():
    prompt = [PAGE_ID] * 2 + [100]
    guard = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID, max_page_tokens=3)

    # Page cap hit while pages remain: only <PAGE> stays finite.
    logits = _step(guard, prompt, [10, 11, 12])
    finite = torch.isfinite(logits).nonzero().flatten().tolist()
    assert finite == [PAGE_ID]

    # Page cap hit on the last page: only EOS stays finite.
    guard = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID, max_page_tokens=3)
    logits = _step(guard, [PAGE_ID, 100], [PAGE_ID, 11, 12, 13])
    finite = torch.isfinite(logits).nonzero().flatten().tolist()
    assert finite == [EOS_ID]


def test_extra_page_markers_force_eos():
    prompt = [PAGE_ID, 100]
    guard = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID)
    # One marker completes the single expected page; the second is extra.
    logits = _step(guard, prompt, [10, PAGE_ID, 11, PAGE_ID])
    finite = torch.isfinite(logits).nonzero().flatten().tolist()
    assert finite == [EOS_ID]


def test_no_prompt_markers_is_passthrough():
    guard = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID, max_page_tokens=1)
    logits = _logits()
    out = guard([100, 101], [10, 11, 12], logits)
    assert out is logits
    assert torch.isfinite(out).all()


def test_nonpositive_page_cap_disables_only_the_cap():
    prompt = [PAGE_ID] * 2 + [100]
    guard = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID, max_page_tokens=0)
    logits = _step(guard, prompt, [10] * 100)
    assert logits[EOS_ID].item() == -float("inf")
    assert torch.isfinite(logits).sum().item() == VOCAB_SIZE - 1


def test_incremental_scanning_matches_one_shot():
    prompt = [PAGE_ID] * 3 + [100]
    output_ids = [10, PAGE_ID, 11, 12, PAGE_ID, 13]

    one_shot = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID, max_page_tokens=5)
    expected = _step(one_shot, prompt, output_ids)

    incremental = PageCountGuardLogitsProcessor(PAGE_ID, EOS_ID, max_page_tokens=5)
    running: list[int] = []
    for token_id in output_ids:
        running.append(token_id)
        actual = _step(incremental, prompt, running)
    assert torch.equal(actual, expected)


def _make_adapter(model: str) -> PageCountGuardPerReqLogitsProcessor:
    vllm_config = VllmConfig(model_config=ModelConfig(model=model))
    return PageCountGuardPerReqLogitsProcessor(vllm_config, torch.device("cpu"), False)


def test_adapter_inactive_without_page_token():
    # A model without sparse_attention_config / extra_args gets no processor.
    adapter = _make_adapter("facebook/opt-125m")
    assert adapter.new_req_logits_processor(SamplingParams()) is None


def test_adapter_resolves_defaults_from_hf_config():
    adapter = _make_adapter("baidu/Unlimited-OCR-2")
    req_lp = adapter.new_req_logits_processor(SamplingParams())
    assert isinstance(req_lp, PageCountGuardLogitsProcessor)
    assert req_lp.page_token_id == 128827
    assert req_lp.eos_token_id == 1
    assert req_lp.max_page_tokens == 20000


def test_adapter_extra_args_override_defaults():
    adapter = _make_adapter("baidu/Unlimited-OCR-2")
    params = SamplingParams(
        extra_args={
            "page_token_id": 5,
            "eos_token_id": 6,
            "max_page_tokens": 42,
        }
    )
    req_lp = adapter.new_req_logits_processor(params)
    assert isinstance(req_lp, PageCountGuardLogitsProcessor)
    assert (req_lp.page_token_id, req_lp.eos_token_id, req_lp.max_page_tokens) == (
        5,
        6,
        42,
    )


@pytest.mark.parametrize(
    "extra_args",
    [
        {"page_token_id": -1},
        {"page_token_id": "<PAGE>"},
        {"eos_token_id": 1.5},
        {"max_page_tokens": "20000"},
    ],
)
def test_adapter_validate_params_rejects_bad_types(extra_args):
    params = SamplingParams(extra_args=extra_args)
    with pytest.raises(ValueError):
        PageCountGuardPerReqLogitsProcessor.validate_params(params)


def test_adapter_apply_through_batch_update():
    adapter = _make_adapter("baidu/Unlimited-OCR-2")
    params = SamplingParams(extra_args={"max_page_tokens": 2})
    prompt_ids = [PAGE_ID, 100]
    output_ids = [10, 11]
    adapter.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, params, prompt_ids, output_ids)],
            moved=[],
        )
    )
    logits = torch.zeros(1, VOCAB_SIZE)
    adapter.apply(logits)
    # Page cap reached with the page unfinished: only <PAGE> stays finite.
    finite = torch.isfinite(logits[0]).nonzero().flatten().tolist()
    assert finite == [PAGE_ID]

    # Completing the page unmasks the row through the live output-ids list.
    output_ids.append(PAGE_ID)
    logits = torch.zeros(1, VOCAB_SIZE)
    adapter.apply(logits)
    assert torch.isfinite(logits[0]).all()


_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Model Runner V2 processors need CUDA buffers",
)


def _make_v2_req_state(max_num_reqs: int = 1, max_len: int = 64):
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsProcRequestState,
    )

    device = torch.device("cuda:0")
    return LogitsProcRequestState(
        device=device,
        max_num_reqs=max_num_reqs,
        vocab_size=VOCAB_SIZE,
        all_token_ids=StagedWriteTensor(
            (max_num_reqs, max_len), dtype=torch.int32, device=device
        ),
        prompt_len=UvaBackedTensor(max_num_reqs, dtype=torch.int32),
        prefill_len=UvaBackedTensor(max_num_reqs, dtype=torch.int32),
        total_len=StagedWriteTensor(max_num_reqs, dtype=torch.int32, device=device),
    )


def _make_v2_ctx(num_reqs: int, seq_lens: list[int]):
    import numpy as np

    from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext

    device = torch.device("cuda:0")
    idx = torch.arange(num_reqs, device=device)
    return LogitsContext(
        expanded_idx_mapping=idx,
        idx_mapping=idx,
        idx_mapping_np=np.arange(num_reqs),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.long, device=device),
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        pos=torch.zeros(num_reqs, dtype=torch.long, device=device),
        seq_lens_upper_bound_np=np.array(seq_lens, dtype=np.int32),
    )


def _make_v2_guard(model: str = "baidu/Unlimited-OCR-2"):
    from vllm.model_executor.models.unlimited_ocr_page_guard import (
        PageCountGuardV2LogitsProcessor,
    )

    vllm_config = VllmConfig(model_config=ModelConfig(model=model))
    return PageCountGuardV2LogitsProcessor(vllm_config, _make_v2_req_state())


def _commit(guard, prompt_ids: list[int], output_ids: list[int]) -> None:
    ids = prompt_ids + output_ids
    guard.prompt_len_np[0] = len(prompt_ids)
    guard.all_token_ids.gpu[0, : len(ids)] = torch.tensor(
        ids, dtype=torch.int32, device=guard.all_token_ids.gpu.device
    )


@_requires_cuda
@pytest.mark.parametrize("max_page_tokens", [0, 3, 20000])
def test_v2_guard_matches_v1_step_by_step(max_page_tokens):
    """The V2 batched guard reproduces the V1 per-request guard each step."""
    prompt_ids = [PAGE_ID, PAGE_ID, 100]
    output_ids = [10, 11, PAGE_ID, 12, 13, 14, PAGE_ID, 15, PAGE_ID]
    v1 = PageCountGuardLogitsProcessor(PAGE_ID, 1, max_page_tokens)

    v2 = _make_v2_guard()
    params = SamplingParams(
        extra_args={
            "page_token_id": PAGE_ID,
            "eos_token_id": 1,
            "max_page_tokens": max_page_tokens,
        },
    )
    assert v2.add_request(0, params)

    running: list[int] = []
    for token_id in output_ids:
        running.append(token_id)
        expected = v1(prompt_ids, running, _logits())
        _commit(v2, prompt_ids, running)
        ctx = _make_v2_ctx(1, [len(prompt_ids) + len(running)])
        actual = v2.apply(_logits().unsqueeze(0).to("cuda"), ctx)[0].cpu()
        assert torch.equal(actual, expected)


@_requires_cuda
def test_v2_guard_inactive_without_page_token():
    v2 = _make_v2_guard("facebook/opt-125m")
    assert not v2.add_request(0, SamplingParams())
    logits = _logits().unsqueeze(0).to("cuda")
    assert v2.apply(logits, _make_v2_ctx(1, [4])) is logits


@_requires_cuda
def test_v2_ngram_matches_v1():
    from vllm.model_executor.models.deepseek_ocr import NoRepeatNGramLogitsProcessor
    from vllm.model_executor.models.unlimited_ocr import NGramV2LogitsProcessor

    prompt_ids = [100, 101]
    output_ids = [5, 6, 7, 8, 5, 6, 7, 9, 5, 6, 7]
    ngram_size, window_size = 3, 8
    v1 = NoRepeatNGramLogitsProcessor(ngram_size, window_size)
    expected = v1(output_ids, _logits())

    vllm_config = VllmConfig(model_config=ModelConfig(model="baidu/Unlimited-OCR-2"))
    v2 = NGramV2LogitsProcessor(vllm_config, _make_v2_req_state())
    params = SamplingParams(
        extra_args={"ngram_size": ngram_size, "window_size": window_size},
    )
    assert v2.add_request(0, params)
    v2.prompt_len_np[0] = len(prompt_ids)
    ids = prompt_ids + output_ids
    v2.all_token_ids.gpu[0, : len(ids)] = torch.tensor(
        ids, dtype=torch.int32, device=v2.all_token_ids.gpu.device
    )
    actual = v2.apply(_logits().unsqueeze(0).to("cuda"), _make_v2_ctx(1, [len(ids)]))[
        0
    ].cpu()
    assert torch.equal(actual, expected)
