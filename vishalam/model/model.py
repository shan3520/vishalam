"""Vishalam decoder-only transformer, written from scratch.

Llama-style: pre-norm RMSNorm, rotary position embeddings (RoPE), SwiGLU feed-forward,
no biases, tied input/output embeddings, PyTorch SDPA attention, optional KV cache for
fast generation and optional activation checkpointing for training memory.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, hd); cos/sin: (T, hd/2). Rotate pairs (first half, second half).
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class KVCache:
    """Preallocated per-layer key/value cache for incremental decoding."""

    def __init__(self, cfg: ModelConfig, batch: int, max_len: int, device, dtype):
        shape = (batch, cfg.n_heads, max_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.n_layers)]
        self.max_len = max_len
        self.pos = 0  # number of positions already filled

    def truncate(self, pos: int) -> None:
        self.pos = pos


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None, layer: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            start = cache.pos
            cache.k[layer][:, :, start : start + T] = k
            cache.v[layer][:, :, start : start + T] = v
            k_all = cache.k[layer][:, :, : start + T]
            v_all = cache.v[layer][:, :, : start + T]
            if T == 1:
                y = F.scaled_dot_product_attention(q, k_all, v_all)
            else:
                # query i (absolute pos start+i) may attend keys 0..start+i
                qpos = torch.arange(start, start + T, device=x.device)[:, None]
                kpos = torch.arange(start + T, device=x.device)[None, :]
                y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=kpos <= qpos)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0):
        x = x + self.attn(self.norm1(x), cos, sin, cache, layer)
        return x + self.mlp(self.norm2(x))


class Vishalam(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.grad_checkpointing = False
        self.apply(self._init)
        # scale residual-branch output projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        # PyTorch/XLA rebuilds every parameter on .to(device), which would untie the output head from
        # the embedding (a different model, and checkpoints that no longer load across GPU/TPU)
        if self.cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        return self

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        z_loss: float = 0.0,
        return_logits: bool = False,
    ):
        """idx: (B, T) token ids. targets: (B, T) next-token ids (or None).
        loss_mask: (B, T) 1 where the target should count toward the loss (e.g. agent tokens only).
        Returns (logits, loss). With targets, logits are None unless return_logits=True (training uses a
        memory-saving chunked loss). Without targets (inference), only the last position's logits are
        computed.
        """
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        cos = self.rope_cos[start : start + T]
        sin = self.rope_sin[start : start + T]
        x = self.embed(idx)
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and self.training and cache is None:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin, cache, i)
        x = self.norm(x)
        if cache is not None:
            cache.pos = start + T

        if targets is None:
            return self.lm_head(x[:, -1:]).float(), None

        flat_targets = targets.reshape(-1)
        if loss_mask is None:
            weights = torch.ones(flat_targets.shape, device=x.device, dtype=torch.float32)
        else:
            weights = loss_mask.reshape(-1).to(torch.float32)
        denom = weights.sum().clamp(min=1.0)

        if return_logits:
            logits = self.lm_head(x).float()
            ce_sum, z_sum = _loss_terms(logits.view(-1, logits.size(-1)), flat_targets, weights, z_loss)
        else:
            # Chunked loss: never materialise (B*T, vocab) fp32 logits at once. Each chunk's logits are
            # recomputed in backward (checkpoint), so peak memory is one chunk instead of the whole batch.
            # Chunks run along the sequence (all rows of the batch together): with the batch split
            # across TPU chips, every chip then only touches its own rows.
            logits = None
            ce_sum = z_sum = torch.zeros((), device=x.device, dtype=torch.float32)
            w = weights.view(B, T)
            span = max(1, LOSS_CHUNK // B)
            for s in range(0, T, span):
                part = (x[:, s : s + span].reshape(-1, x.size(-1)), targets[:, s : s + span].reshape(-1),
                        w[:, s : s + span].reshape(-1), z_loss)
                if torch.is_grad_enabled():
                    c, z = checkpoint(self._chunk_loss, *part, use_reentrant=False)
                else:
                    c, z = self._chunk_loss(*part)
                ce_sum, z_sum = ce_sum + c, z_sum + z
        loss = ce_sum / denom
        if z_loss > 0:
            loss = loss + z_loss * z_sum / denom
        return logits, loss

    def _chunk_loss(self, h, t, w, z_loss: float):
        return _loss_terms(self.lm_head(h).float(), t, w, z_loss)


LOSS_CHUNK = 4096  # tokens per loss chunk (~0.8 GB of fp32 logits for a 50k vocab)


def _loss_terms(flat_logits, targets, weights, z_loss: float):
    ce = F.cross_entropy(flat_logits, targets, reduction="none")
    ce_sum = (ce * weights).sum()
    if z_loss > 0:
        lse = torch.logsumexp(flat_logits, dim=-1)
        z_sum = ((lse**2) * weights).sum()
    else:
        z_sum = torch.zeros((), device=flat_logits.device, dtype=torch.float32)
    return ce_sum, z_sum
