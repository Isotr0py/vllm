# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Unlimited-OCR (baidu/Unlimited-OCR) reuses
# the DeepSeek-OCR multimodal layout (DeepEncoder = SAM-ViT-B + CLIP-L, a linear
# MLP projector and a DeepSeek-V2 text backbone). The only architectural
# difference is the language model, which is a DeepSeek-V2 *MoE* with plain
# multi-head attention (``use_mla=False``) instead of the dense MLA backbone.
# We therefore reuse ``DeepseekVLV2Config`` for parsing the nested config.

from typing import Any

from transformers import PreTrainedConfig

from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config


class UnlimitedOCRSparseAttentionConfig(PreTrainedConfig):
    """SR-SWA sparse attention sub-config of Unlimited-OCR-2.

    Mirrors the ``sparse_attention_config`` dict in the reference
    ``config.json`` (``baidu/Unlimited-OCR-2``), including the DASH router
    knobs and the ``page2d`` sparse RoPE parameters. Unlimited-OCR (v1)
    configs do not carry this section and keep
    ``UnlimitedOCRConfig.sparse_attention_config is None``.
    """

    model_type = "unlimited-ocr-sparse-attention"

    def __init__(
        self,
        use_sparse_attention: bool = True,
        sparse_block_size: int = 32,
        sparse_topk_blocks: int = 16,
        sparse_index_dim: int = 128,
        sparse_num_index_heads: int = 1,
        sparse_init_block: int = 1,
        sparse_local_block: int = 4,
        sparse_attention_freq: list[int] | None = None,
        sparse_rope_mode: str = "page2d",
        sparse_rope_theta: float = 500000.0,
        sparse_rope_tok_freq_mode: str = "resample",
        sparse_rope_tok_theta: float = 10000.0,
        sparse_rope_page_mod: int = 2,
        sparse_rope_page_pairs: int = 1,
        sparse_rope_page_span: int = 0,
        sparse_rope_page_theta: float = 256.0,
        sparse_main_rope_mode: str = "page2d",
        sparse_main_rope_tok_freq_mode: str = "resample",
        dash_router: str = "entmax_topk_perlayer",
        dash_topk_blocks: int = 12,
        dash_scaling_factor: float = 128.0,
        dash_scale_per_layer: list[float] | None = None,
        dash_floor_per_layer: list[float] | None = None,
        dash_sigma: float = 1.0,
        dash_prior_floor: float = 0.0001,
        dash_stage0: str = "head_cls",
        dash_query_chunk: int = 2048,
        dash_image_only: bool = True,
        dash_use_triton: bool = True,
        image_bucket_constraint: bool = True,
        page_token_id: int = 128827,
        image_token_id: int = 128815,
        per_page_image_tokens: int = 273,
        training_stage: str = "dash",
        use_flex_attention: bool = False,
        use_xformers: bool = False,
        use_kl_loss: bool = False,
        kl_loss_weight: float = 1.0,
        kl_chunk_size: int = 1024,
        kl_log_interval: int = 1,
        **kwargs,
    ):
        self.use_sparse_attention = use_sparse_attention
        self.sparse_block_size = sparse_block_size
        self.sparse_topk_blocks = sparse_topk_blocks
        self.sparse_index_dim = sparse_index_dim
        self.sparse_num_index_heads = sparse_num_index_heads
        self.sparse_init_block = sparse_init_block
        self.sparse_local_block = sparse_local_block
        self.sparse_attention_freq = sparse_attention_freq
        self.sparse_rope_mode = sparse_rope_mode
        self.sparse_rope_theta = sparse_rope_theta
        self.sparse_rope_tok_freq_mode = sparse_rope_tok_freq_mode
        self.sparse_rope_tok_theta = sparse_rope_tok_theta
        self.sparse_rope_page_mod = sparse_rope_page_mod
        self.sparse_rope_page_pairs = sparse_rope_page_pairs
        self.sparse_rope_page_span = sparse_rope_page_span
        self.sparse_rope_page_theta = sparse_rope_page_theta
        self.sparse_main_rope_mode = sparse_main_rope_mode
        self.sparse_main_rope_tok_freq_mode = sparse_main_rope_tok_freq_mode
        self.dash_router = dash_router
        self.dash_topk_blocks = dash_topk_blocks
        self.dash_scaling_factor = dash_scaling_factor
        self.dash_scale_per_layer = dash_scale_per_layer
        self.dash_floor_per_layer = dash_floor_per_layer
        self.dash_sigma = dash_sigma
        self.dash_prior_floor = dash_prior_floor
        self.dash_stage0 = dash_stage0
        self.dash_query_chunk = dash_query_chunk
        self.dash_image_only = dash_image_only
        self.dash_use_triton = dash_use_triton
        self.image_bucket_constraint = image_bucket_constraint
        self.page_token_id = page_token_id
        self.image_token_id = image_token_id
        self.per_page_image_tokens = per_page_image_tokens
        self.training_stage = training_stage
        self.use_flex_attention = use_flex_attention
        self.use_xformers = use_xformers
        self.use_kl_loss = use_kl_loss
        self.kl_loss_weight = kl_loss_weight
        self.kl_chunk_size = kl_chunk_size
        self.kl_log_interval = kl_log_interval

        super().__init__(**kwargs)


class UnlimitedOCRConfig(DeepseekVLV2Config):
    model_type = "unlimited-ocr"

    # An explicit ``__init__`` is required: Transformers v5 processes each
    # concrete config class' ``__init__`` signature to build nested sub-configs,
    # and an empty subclass (only overriding ``model_type``) would skip
    # ``DeepseekVLV2Config.__init__``, leaving ``text_config`` unset.
    def __init__(
        self,
        tile_tag: str = "2D",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),),
        rswa_window: int = 128,
        max_position_embeddings: int = 32768,
        sparse_attention_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        # Set before ``super().__init__``: Transformers v5 RoPE standardization
        # (triggered by the top-level YaRN ``rope_scaling`` of Unlimited-OCR-2
        # configs) reads it during ``PreTrainedConfig.__post_init__``.
        self.max_position_embeddings = max_position_embeddings
        super().__init__(
            tile_tag=tile_tag,
            global_view_pos=global_view_pos,
            candidate_resolutions=candidate_resolutions,
            **kwargs,
        )
        self.rswa_window = (
            # Unlimited-OCR-2 replaces R-SWA with DASH SR-SWA; v1 keeps the
            # windowed attention.
            None if sparse_attention_config is not None else rswa_window
        )
        # Unlimited-OCR-2 only; v1 configs leave this as None.
        self.sparse_attention_config = (
            UnlimitedOCRSparseAttentionConfig(**sparse_attention_config)
            if sparse_attention_config is not None
            else None
        )
