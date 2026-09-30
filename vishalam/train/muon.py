"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz), written from scratch.

Idea (Keller Jordan, modded-nanogpt): for a 2-D weight matrix, take the momentum-smoothed gradient
and replace it by the nearest (semi-)orthogonal matrix, i.e. keep its "directions" but set all
singular values to ~1. That update is computed with a few Newton-Schulz iterations. It is used only
for the hidden 2-D matrices of the transformer; embeddings, the (tied) output head and all 1-D
parameters (norms) keep using AdamW.
"""
from __future__ import annotations

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # quintic iteration tuned to converge fast from a normalized start


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a 2-D matrix: returns ~U @ V^T for G = U S V^T.
    Runs in fp32 (T4 GPUs have no fast bf16; the cost is tiny next to the forward/backward)."""
    a, b, c = NS_COEFFS
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)  # spectral norm <= Frobenius norm <= 1, so the iteration converges
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters.

    Param-group options: lr, momentum, nesterov, weight_decay (decoupled), ns_steps, and
    split (orthogonalize a fused matrix in `split` equal row-blocks, e.g. 3 for a fused q/k/v).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = 5, split: int = 1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                        ns_steps=ns_steps, split=split)
        super().__init__(params, defaults)
        for g in self.param_groups:
            for p in g["params"]:
                if p.dim() != 2:
                    raise ValueError("Muon only handles 2-D parameters; use AdamW for the rest")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, mom, split = g["lr"], g["momentum"], g["split"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - mom)                          # buf = mom*buf + (1-mom)*grad
                update = grad.lerp(buf, mom) if g["nesterov"] else buf
                if split > 1:
                    parts = update.chunk(split, dim=0)
                    ortho = torch.cat([newton_schulz(u, g["ns_steps"]) for u in parts], dim=0)
                    rows = p.size(0) // split
                else:
                    ortho = newton_schulz(update, g["ns_steps"])
                    rows = p.size(0)
                scale = max(1.0, rows / p.size(1)) ** 0.5         # keep update RMS comparable across shapes
                if g["weight_decay"]:
                    p.mul_(1 - lr * g["weight_decay"])
                p.add_(ortho.to(p.dtype), alpha=-lr * scale)
        return loss
