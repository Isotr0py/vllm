# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DASH router for baidu/Unlimited-OCR-2 SR-SWA attention.

Framework-agnostic inference port of the Unlimited-OCR-2 HF reference
(``dash_attention.py`` and the routing helpers of ``modeling_deepseekv2.py``):
1.5-entmax over block scores (sorted closed-form threshold), a learned
``head_cls`` block summary, and the inference route select -- 1.5-entmax on
the top-M block-score window, hard top-K read set, mean-centered log-prior
bias ``(log(clamp(p, floor)) - mu) / sigma``.

Training-only pieces of the reference (the ``_KSchedule`` k-anneal and the
``_PerLayerScheduler`` temperature closed loop) are deliberately not ported.
Inference reproduces the trained routing temperature from the per-layer
``dash_scale_per_layer`` / ``dash_floor_per_layer`` arrays the checkpoint
exports; see :meth:`DashRouter.set_inference_params`.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Finite "not a remote block" sentinel for block scores (never a real -inf).
_SCORE_NEG = -1.0e9
# Finite mask fill for entmax candidate masking.
_ENTMAX_NEG = -1.0e4
# Activation bound used by the reference router on every score path.
_ACT_CLAMP = 1.0e4


def _make_ix_like(x: torch.Tensor, dim: int) -> torch.Tensor:
    d = x.size(dim)
    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _entmax_threshold(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Closed-form 1.5-entmax threshold tau* along ``dim``."""
    x_srt, _ = torch.sort(x, descending=True, dim=dim)
    rho = _make_ix_like(x, dim)
    mean = x_srt.cumsum(dim) / rho
    mean_sq = (x_srt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho
    tau = mean - torch.sqrt(torch.clamp(delta, 0))
    support_size = (tau <= x_srt).sum(dim).unsqueeze(dim).clamp(min=1)
    return tau.gather(dim, support_size - 1)


def entmax15(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1.5-entmax (Peters, Niculae & Martins, 2019), forward only.

    Masked-out entries must be passed as a finite, very-negative value (not
    -inf) so the sorting-based threshold stays numerically well-defined; they
    receive probability exactly 0. Matches the reference ``entmax15`` in
    ``dash_attention.py`` bit-for-bit on the forward path.
    """
    max_val = x.amax(dim=dim, keepdim=True)
    x = (x - max_val) / 2
    tau_star = _entmax_threshold(x, dim)
    return torch.clamp(x - tau_star, min=0) ** 2


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard Llama rotary application (transformers parity)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm (scale = 1 + weight, weight zero-init).

    Same fp32 math and finiteness guards as the reference router norms; the
    module name and parameter layout match the checkpoint (``weight``).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        x32 = x32.clamp(-_ACT_CLAMP, _ACT_CLAMP)
        out = (x32 * (1.0 + self.weight.float())).clamp(-_ACT_CLAMP, _ACT_CLAMP)
        return out.to(x.dtype)


@dataclass
class DashRouteParams:
    """Frozen per-layer routing parameters for inference.

    Attributes:
        idx_scale: entmax input scale, ``dash_scale_per_layer[layer] /
            sqrt(index_dim)``.
        floor: prior floor, ``dash_floor_per_layer[layer]``.
        sigma: log-prior bias divisor (``dash_sigma``).
        topk: hard read budget K (``dash_topk_blocks``, 12 for Unlimited-OCR-2).
        topm: entmax window M (deploy default 16); <= 0 means all blocks.

    """

    idx_scale: float
    floor: float
    sigma: float
    topk: int
    topm: int


class DashRouter(nn.Module):
    """DASH block router: index projections, block scoring, entmax routing.

    Parameter names (``q_proj``, ``k_proj``, ``q_norm``, ``k_norm``,
    ``head_cls``) match the HF reference checkpoint one-to-one, so the router
    weights can be loaded with a plain ``load_state_dict`` per layer.

    All methods are shape-static given (batch, query length, block count) and
    data-independent in control flow, so they are safe under CUDA graphs.
    """

    def __init__(
        self,
        hidden_size: int,
        index_dim: int,
        block_size: int,
        local_blocks: int,
        sigma: float = 1.0,
        rms_norm_eps: float = 1e-6,
        stage0: str = "head_cls",
        q_chunk: int = 2048,
    ):
        super().__init__()
        if stage0 not in ("head_cls", "indexer"):
            raise ValueError(f"unknown DASH stage0 mode: {stage0!r}")
        self.index_dim = index_dim
        self.block_size = block_size
        self.local_blocks = max(1, local_blocks)
        self.local_window = self.local_blocks * block_size
        self.sigma = float(sigma)
        self.stage0 = stage0
        self.q_chunk = q_chunk

        self.q_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.q_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(index_dim, eps=rms_norm_eps)
        self.head_cls = nn.Parameter(torch.zeros(index_dim))

        self.route_params = DashRouteParams(
            idx_scale=1.0 / math.sqrt(index_dim),
            floor=1e-2,
            sigma=self.sigma,
            topk=12,
            topm=16,
        )

    def set_inference_params(
        self,
        idx_scale: float,
        floor: float,
        topk: int,
        topm: int,
        sigma: float | None = None,
    ) -> None:
        """Install the per-layer routing parameters exported by the checkpoint.

        Args:
            idx_scale: ``dash_scale_per_layer[layer] / sqrt(index_dim)``.
            floor: ``dash_floor_per_layer[layer]``.
            topk: ``dash_topk_blocks`` (hard read budget K).
            topm: entmax window M (deploy default 16; <= 0 = all blocks).
            sigma: ``dash_sigma``; keeps the constructor value when None.

        """
        self.route_params = DashRouteParams(
            idx_scale=float(idx_scale),
            floor=float(floor),
            sigma=self.sigma if sigma is None else float(sigma),
            topk=int(topk),
            topm=int(topm),
        )

    def index_qk(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states into the single-head index space.

        Args:
            hidden_states: [B, S, hidden_size].
            position_embeddings: optional (cos, sin) [B, S, D] with
                D >= index_dim; the first ``index_dim`` channels are applied
                to the index q/k exactly as the reference does.

        Returns:
            (idx_q, idx_k), each [B, S, index_dim] fp32.

        """
        B, S, _ = hidden_states.shape
        hs = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        idx_q = self.q_norm(self.q_proj(hs).view(B, S, 1, self.index_dim))
        idx_k = self.k_norm(self.k_proj(hs).view(B, S, 1, self.index_dim))
        idx_q = idx_q.transpose(1, 2)
        idx_k = idx_k.transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            idx_q, idx_k = apply_rotary_pos_emb(
                idx_q,
                idx_k,
                cos[..., : self.index_dim],
                sin[..., : self.index_dim],
            )
        idx_q = torch.nan_to_num(idx_q[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        idx_k = torch.nan_to_num(idx_k[:, 0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        return idx_q, idx_k

    def head_cls_summary(
        self,
        idx_k: torch.Tensor,
        nblk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage-0 block summary: head_cls masked-softmax over each block.

        A per-router learned summary query ``head_cls`` (zero-init => mean
        pooling) attends the index keys within each block to form one summary
        key ``k_bar_c``.

        Args:
            idx_k: [B, S, index_dim] fp32 index keys (RoPE'd).
            nblk: number of KV blocks (S is padded up to ``nblk * block_size``).

        Returns:
            (k_bar [B, nblk, index_dim] fp32, chunk_valid [B, nblk] bool).

        """
        B, S, d = idx_k.shape
        bs = self.block_size
        padded = nblk * bs
        if padded > S:
            kk = F.pad(idx_k, (0, 0, 0, padded - S))
            real = F.pad(
                idx_k.new_ones(B, S, dtype=torch.bool), (0, padded - S), value=False
            )
        else:
            kk = idx_k[:, :padded]
            real = idx_k.new_ones(B, padded, dtype=torch.bool)
        kk = kk.view(B, nblk, bs, d)
        vmask = real.view(B, nblk, bs)

        inv_sqrt_d = 1.0 / math.sqrt(d)
        hc = self.head_cls.to(kk.dtype).view(1, 1, 1, d)
        aw = ((kk * hc).sum(-1) * inv_sqrt_d).clamp(-_ACT_CLAMP, _ACT_CLAMP)
        # NaN-safe masked softmax: valid-only row max, multiplicative padding.
        neg = torch.finfo(aw.dtype).min
        row_max = torch.where(vmask, aw, aw.new_full((), neg)).amax(
            dim=-1, keepdim=True
        )
        row_max = torch.where(
            vmask.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max)
        )
        e = torch.exp(aw - row_max) * vmask.to(aw.dtype)
        w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        k_bar = (w.unsqueeze(-1) * kk).sum(dim=2)
        k_bar = torch.nan_to_num(k_bar, nan=0.0, posinf=0.0, neginf=0.0)
        return k_bar, vmask.any(dim=-1)

    def block_scores(
        self,
        idx_q: torch.Tensor,
        query_positions: torch.Tensor,
        nblk: int,
        idx_k: torch.Tensor | None = None,
        k_bar: torch.Tensor | None = None,
        chunk_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Remote-only block scores [B, Lq, nblk] fp32.

        ``stage0 == "head_cls"``: ``z[i, c] = idx_q_i . k_bar_c / sqrt(d)``
        (pass ``k_bar``/``chunk_valid`` from :meth:`head_cls_summary`).
        ``stage0 == "indexer"``: block-max of the token-level ``idx_q .
        idx_k`` (pass ``idx_k``, the full cached index keys). A block is a
        remote candidate iff it lies fully before the local window; all other
        entries are filled with ``_SCORE_NEG``.

        Args:
            idx_q: [B, Lq, index_dim] fp32 index queries.
            query_positions: [B, Lq] long flat token positions of the queries.
            nblk: number of KV blocks.
            idx_k: [B, S, index_dim] fp32 cached index keys (indexer mode).
            k_bar: [B, nblk, index_dim] fp32 block summaries (head_cls mode).
            chunk_valid: [B, nblk] bool blocks holding >= 1 real token.

        """
        B, Lq, d = idx_q.shape
        bs = self.block_size
        inv_sqrt_d = 1.0 / math.sqrt(d)
        blk_ids = torch.arange(nblk, device=idx_q.device)
        out = []
        if self.stage0 == "head_cls":
            if k_bar is None or chunk_valid is None:
                raise ValueError("head_cls scoring needs k_bar/chunk_valid")
            for c0 in range(0, Lq, self.q_chunk):
                c1 = min(c0 + self.q_chunk, Lq)
                z = torch.matmul(idx_q[:, c0:c1], k_bar.transpose(-1, -2)) * inv_sqrt_d
                z = z.clamp(-_ACT_CLAMP, _ACT_CLAMP)
                qblk = query_positions[:, c0:c1] // bs
                keep = blk_ids[None, None, :] <= (qblk[:, :, None] - self.local_blocks)
                keep = keep & chunk_valid[:, None, :]
                out.append(z.masked_fill(~keep, _SCORE_NEG))
            return torch.cat(out, dim=1)

        if idx_k is None:
            raise ValueError("indexer scoring needs idx_k")
        S = idx_k.shape[1]
        kpos = torch.arange(S, device=idx_k.device)
        for c0 in range(0, Lq, self.q_chunk):
            c1 = min(c0 + self.q_chunk, Lq)
            sc = torch.matmul(idx_q[:, c0:c1], idx_k.transpose(-1, -2))
            sc = sc.clamp(-_ACT_CLAMP, _ACT_CLAMP)
            pos = query_positions[:, c0:c1]
            future = kpos[None, None, :] > pos[:, :, None]
            local = (
                kpos[None, None, :] > (pos[:, :, None] - self.local_window)
            ) & ~future
            sc = sc.masked_fill(future | local, _SCORE_NEG)
            padded = nblk * bs
            if padded > S:
                sc = F.pad(sc, (0, padded - S), value=_SCORE_NEG)
            out.append(sc.view(B, c1 - c0, nblk, bs).amax(dim=-1))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def route_and_select(
        self,
        block_scores: torch.Tensor,
        block_cand: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference routing: top-M entmax window + hard top-K read set.

        Parity with the reference ``_infer_route_select``: 1.5-entmax runs on
        the top-M candidate scores only, the read set is the first K candidate
        slots of that window, and the bias is the mean-centered log-prior
        ``(log(clamp(p, floor)) - mu) / sigma`` with ``mu`` the per-query mean
        over the kept log-priors.

        Args:
            block_scores: [B, Lq, nblk] fp32; ``_SCORE_NEG`` = not a remote
                block.
            block_cand: [B, Lq, nblk] bool page-aware candidate domain, or
                None for all remote blocks. Rows with no candidate fall back
                to all remote blocks.

        Returns:
            (topi [B, Lq, K] int64 block indices, bias [B, Lq, K] fp32,
            valid [B, Lq, K] bool).

        """
        p = self.route_params
        remote = block_scores > (_SCORE_NEG * 0.5)
        cand = remote if block_cand is None else (remote & block_cand)
        cand = torch.where(~cand.any(dim=-1, keepdim=True), remote, cand)
        x = torch.where(
            cand, block_scores * p.idx_scale, block_scores.new_full((), _ENTMAX_NEG)
        )
        x = torch.nan_to_num(x, nan=_ENTMAX_NEG, posinf=_ENTMAX_NEG, neginf=_ENTMAX_NEG)
        nblk = x.shape[-1]
        m = max(1, min(p.topm, nblk)) if p.topm > 0 else nblk
        xv, xi = x.topk(m, dim=-1)
        prior = entmax15(xv, dim=-1)
        is_cand = cand.gather(-1, xi)
        k = max(1, min(p.topk, m))
        xi, prior, valid = xi[..., :k], prior[..., :k], is_cand[..., :k]
        logp = torch.log(prior.clamp_min(p.floor))
        cnt = valid.sum(dim=-1, keepdim=True).clamp(min=1)
        mu = (
            torch.where(valid, logp, torch.zeros_like(logp)).sum(dim=-1, keepdim=True)
            / cnt
        )
        bias = torch.where(valid, (logp - mu) * (1.0 / p.sigma), torch.zeros_like(logp))
        return xi, bias.float(), valid
