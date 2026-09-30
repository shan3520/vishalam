"""Generates notebooks/colab_tpu_benchmark.ipynb: a self-contained Colab TPU (v5e-1) speed test.

    python notebooks/make_colab_tpu.py

The notebook embeds Vishalam's model/config/Muon source (no GitHub/HF tokens, no private repo access)
and trains on random tokens: speed does not depend on token content. It measures tokens/s for
  * 125M + AdamW  (compare: Kaggle 2xT4 measured 13,700 tok/s)
  * 350M + Muon   (the real target; Kaggle 2xT4 projected ~5,000 tok/s)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["vishalam/__init__.py", "vishalam/model/__init__.py", "vishalam/model/config.py",
         "vishalam/model/model.py", "vishalam/train/__init__.py", "vishalam/train/muon.py"]

INSTALL = '''\
# --- 1. PyTorch/XLA for the TPU (installs a torch build that matches torch_xla) ---
import importlib.util, subprocess, sys
if importlib.util.find_spec("torch_xla") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_xla[tpu]",
                    "-f", "https://storage.googleapis.com/libtpu-releases/index.html"], check=True)
import torch, torch_xla
print("torch", torch.__version__, "| torch_xla", torch_xla.__version__)
'''

BENCH = '''\
# --- 3. Benchmark: tokens/s on the TPU for the real model code ---
import os, statistics, sys, time
sys.path.insert(0, "/content/vishalam_src")
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.checkpoint as xla_ckpt
import vishalam.model.model as vm
from vishalam.model.config import ModelConfig
from vishalam.model.model import Vishalam
from vishalam.train.muon import Muon

# XLA-aware activation checkpointing (the plain torch version can be optimised away by the XLA compiler)
vm.checkpoint = lambda fn, *a, use_reentrant=False: xla_ckpt.checkpoint(fn, *a)
dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
print("device:", dev, xm.xla_device_hw(dev) if hasattr(xm, "xla_device_hw") else "")

SIZES = {
    "125m": dict(vocab_size=50304, d_model=768, n_layers=12, n_heads=12, ffn_dim=2048, max_seq_len=4096),
    "350m": dict(vocab_size=50304, d_model=1024, n_layers=24, n_heads=16, ffn_dim=2816, max_seq_len=4096),
}

def make_opts(model, kind):
    if kind == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), foreach=True)]
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.dim() == 2]
    other = [p for n, p in model.named_parameters() if not (n.startswith("blocks.") and p.dim() == 2)]
    return [Muon(hidden, lr=0.02), torch.optim.AdamW(other, lr=1e-3, betas=(0.9, 0.95), foreach=True)]

def bench(size, kind, micro, ckpt, steps=14, warm=4, seq=4096):
    torch.manual_seed(0)
    model = Vishalam(ModelConfig(**SIZES[size])).to(dev)
    model.grad_checkpointing = ckpt
    opts = make_opts(model, kind)
    x = torch.randint(0, 50304, (micro, seq + 1)).to(dev)
    times, loss = [], None
    for i in range(steps):
        t = time.time()
        with torch.autocast("xla", dtype=torch.bfloat16):
            _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        torch_xla.sync() if hasattr(torch_xla, "sync") else xm.mark_step()
        xm.wait_device_ops()
        times.append(time.time() - t)
        print(f"  step {i}: {times[-1]:.2f}s loss {loss.item():.3f}", flush=True)
    dt = statistics.median(times[warm:])
    tps = micro * seq / dt
    print(f"RESULT {size} {kind} micro={micro} ckpt={ckpt}: {tps:,.0f} tokens/s ({dt:.2f}s/step, first step {times[0]:.0f}s compile)")
    del model, opts
    return tps

results = {}
for size, kind, micros, ckpt in [("125m", "adamw", [4, 2], False), ("350m", "muon", [4, 2, 1], True)]:
    for micro in micros:  # biggest micro-batch that fits the 16 GB HBM
        try:
            results[(size, kind)] = bench(size, kind, micro, ckpt)
            break
        except Exception as e:
            print(f"  {size} micro={micro} failed: {type(e).__name__}: {str(e)[:200]}")

print("\\n==== SUMMARY ====")
k125 = results.get(("125m", "adamw"))
k350 = results.get(("350m", "muon"))
if k125: print(f"125M AdamW : {k125:,.0f} tok/s on v5e-1  vs 13,700 on Kaggle 2xT4  -> {k125 / 13700:.1f}x")
if k350: print(f"350M Muon  : {k350:,.0f} tok/s on v5e-1  -> 10.5B tokens = {10.5e9 / k350 / 3600:,.0f} TPU-hours")
'''


def main() -> None:
    write = ["# --- 2. Vishalam model source (embedded; no repo access needed) ---", "import os"]
    for f in FILES:
        src = (ROOT / f).read_text(encoding="utf-8")
        write.append(f"os.makedirs(os.path.dirname('/content/vishalam_src/{f}'), exist_ok=True)")
        write.append(f"open('/content/vishalam_src/{f}', 'w').write({src!r})")
    write.append('print("source written")')
    cells = [INSTALL, "\n".join(write) + "\n", BENCH]
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "TPU",
                     "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "colab_tpu_benchmark.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
