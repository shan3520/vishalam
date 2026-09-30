"""Full, resumable training checkpoints (atomic writes)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Peel torch.compile (`_orig_mod`) and DDP (`module`) wrappers, in any order, so checkpoints
    always hold the plain model's parameter names."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel) or (
                hasattr(model, "module") and isinstance(getattr(model, "module"), torch.nn.Module)):
            model = model.module
        else:
            return model


def _opt_state(optimizer):
    """One optimizer (or a list of one) is saved exactly as before; several are saved as a list."""
    if isinstance(optimizer, (list, tuple)):
        states = [o.state_dict() for o in optimizer]
        return states[0] if len(states) == 1 else states
    return optimizer.state_dict()


def _load_opt_state(optimizer, saved) -> None:
    opts = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
    saved = saved if isinstance(saved, list) else [saved]
    if len(saved) != len(opts):
        raise ValueError(f"checkpoint has {len(saved)} optimizer(s), this run builds {len(opts)} "
                         "(did the optimizer setting change between sessions?)")
    for opt, sd in zip(opts, saved):
        base = [g.get("base_lr") for g in opt.param_groups]
        opt.load_state_dict(sd)
        for g, b in zip(opt.param_groups, base):  # older checkpoints predate base_lr
            if b is not None:
                g["base_lr"] = b


def save_checkpoint(path: str | Path, model, optimizer, scaler, step: int, extra: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap(model).state_dict(),
        "optimizer": _opt_state(optimizer),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "extra": extra,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, model, optimizer=None, scaler=None, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    unwrap(model).load_state_dict(state["model"])
    if optimizer is not None:
        _load_opt_state(optimizer, state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    rng = state.get("rng") or {}
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            except RuntimeError:
                pass  # different GPU count (e.g. Kaggle 2 GPUs -> Colab 1 GPU)
    return state


def save_weights(path: str | Path, model, model_config: dict, dtype=torch.float16) -> Path:
    """Weights-only export for inference / fine-tuning (half the size, no optimizer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in unwrap(model).state_dict().items()}
    torch.save({"model": sd, "config": model_config}, path)
    return path
