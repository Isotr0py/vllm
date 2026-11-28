# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fill a daemon's model with a source replica's post-processed weights."""

import contextlib

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    build_model_from_processed_weights,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    WeightCacheKey,
    check_weight_cache_loader_support,
)
from vllm.model_executor.model_loader.weight_cache.transport import SeedTransport
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    MOONCAKE_BACKEND,
    PEER_IPC_BACKEND,
    get_mover,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    validate_weights_payload,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument

logger = init_logger(__name__)


@instrument(span_name="Load model")
def load_peer_model(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    prefix: str = "",
    *,
    cache_key: WeightCacheKey,
    transport: SeedTransport,
) -> nn.Module:
    """Load a daemon's model from a source replica's weight cache daemon."""
    check_weight_cache_loader_support(
        vllm_config, model_config, is_draft=cache_key.is_draft
    )
    backend = (
        PEER_IPC_BACKEND if isinstance(transport.source, str) else MOONCAKE_BACKEND
    )
    raw = transport.fetch("weights", seed_backend=backend)
    try:
        payload = validate_weights_payload(raw, cache_key, model_config.model)
        mover = get_mover(payload["seed"]["backend"])
    except Exception:
        with contextlib.suppress(Exception):
            transport.fetch("weights", acknowledge=True)
        raise
    manifest = payload["manifest"]
    device = torch.device(
        current_platform.device_type, torch.accelerator.current_device_index()
    )
    tensors = mover.fill(manifest, payload["seed"], device)
    transport.fetch("weights", acknowledge=True)
    model = build_model_from_processed_weights(
        vllm_config,
        model_config,
        prefix,
        tensors,
        manifest,
        aliases=dict(payload["aliases"]),
        attrs=dict(payload["attrs"]),
        load_config=vllm_config.load_config,
    )
    logger.info(
        "Loaded %d tensors from a peer daemon over %s",
        len(tensors),
        payload["seed"]["backend"],
    )
    return model
