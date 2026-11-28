# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fill a daemon's model with a source replica's post-processed weights."""

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    build_model_from_processed_weights,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    check_weight_cache_loader_support,
)
from vllm.model_executor.model_loader.weight_cache.transport import SeedTransport
from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    acknowledge_lease,
    get_mover,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    WeightsSeedPayload,
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
    is_draft: bool = False,
    payload: WeightsSeedPayload,
    transport: SeedTransport,
) -> nn.Module:
    """Load a daemon's model from a source replica's weight cache daemon."""
    check_weight_cache_loader_support(vllm_config, model_config, is_draft=is_draft)
    manifest = payload["manifest"]
    mover = get_mover(payload["seed"]["backend"])
    device = torch.device(
        current_platform.device_type, torch.accelerator.current_device_index()
    )
    tensors = mover.fill(manifest, payload["seed"], device)
    acknowledge_lease(transport)
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
