# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backends for the Unlimited-OCR-2 DASH layers.

DASH attention does not run any stock attention kernel: the model layer
writes the paged K/V and router-index caches itself and launches the
bias-aware block-sparse Triton kernels directly (see
:mod:`vllm.model_executor.models.unlimited_ocr2.attention`). The backends
here exist only so the v1 runner can group the layers, allocate their KV
cache tensors, and build per-group metadata. The metadata is the plain
:class:`CommonAttentionMetadata` (block table, slot mapping, seq lens),
which is all the DASH layer consumes.

Two backend classes are provided:

* ``DashAttentionBackend`` for the main K/V cache group (one
  ``FullAttentionSpec`` per DASH layer), and
* ``DashIndexBackend`` for the router index-key cache group (one fp32
  single-head "K-only" page per DASH layer).

``DashAttentionBackend`` is registered as a dynamic ``AttentionBackendEnum``
member because ``Attention.__init__`` resolves ``self.attn_backend``'s name
through that enum.
"""

from typing import ClassVar

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

DASH_BACKEND_NAME = "DASH_UNLIMITED_OCR"


class DashAttentionImpl(AttentionImpl):
    """No-op impl: the DASH model layer never calls it."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.supports_quant_query_input = False

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "DashAttentionImpl.forward must never run: the Unlimited-OCR-2 "
            "DASH layer writes the caches and launches its Triton kernels "
            "directly."
        )


class DashMetadataBuilder(AttentionMetadataBuilder[CommonAttentionMetadata]):
    """Pass-through builder: the DASH layer reads CommonAttentionMetadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CommonAttentionMetadata:
        return common_attn_metadata


class DashAttentionBackend(AttentionBackend):
    """Backend marker for the Unlimited-OCR-2 DASH main K/V cache group."""

    @staticmethod
    def get_name() -> str:
        return DASH_BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type[DashAttentionImpl]:
        return DashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[DashMetadataBuilder]:
        return DashMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        # One paged KV block must equal one DASH sparse block (32 tokens);
        # never split pages into smaller kernel blocks.
        if kv_cache_spec is not None:
            return [kv_cache_spec.block_size]
        return [32]


class DashIndexBackend(AttentionBackend):
    """Backend marker for the DASH router index-key cache group."""

    @staticmethod
    def get_name() -> str:
        return f"{DASH_BACKEND_NAME}_INDEX"

    @staticmethod
    def get_impl_cls() -> type[DashAttentionImpl]:
        return DashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[DashMetadataBuilder]:
        return DashMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec=None,
    ) -> list[int | MultipleOf]:
        if kv_cache_spec is not None:
            return [kv_cache_spec.block_size]
        return [32]


# Attention.__init__ resolves backend names through AttentionBackendEnum, so
# the DASH backend needs a dynamic enum member. Registration is idempotent
# across engine restarts in one process.
if DASH_BACKEND_NAME not in AttentionBackendEnum._member_map_:
    AttentionBackendEnum.register(
        DASH_BACKEND_NAME,
        f"{__name__}.DashAttentionBackend",
    )
