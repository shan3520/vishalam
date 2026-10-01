"""Generates notebooks/vishalam_tpu_bench/: a Kaggle TPU v5e-8 (8 chips) speed test for the 350M model.

    python notebooks/make_kaggle_tpu_bench.py

Kaggle gives 20 TPU hours/week on top of the 30 GPU hours. CLI pushes do not get a TPU
(kaggle-cli issue #1197), so after pushing, the run is started from the Kaggle editor with the
accelerator set to "TPU VM v5e-8". Model source is embedded (no secrets, no dataset needed);
training data are random tokens because speed does not depend on token content.

Each config runs in its own process (an XLA out-of-memory error then only loses that config).
8 chips = 8 processes (torch_xla.launch); gradients are averaged with one all_reduce per step,
then every optimizer (Muon + AdamW) steps locally, exactly like DDP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from make_colab_tpu import FILES  # noqa: E402

INSTALL = '''\
# --- 1. PyTorch/XLA (install only if the TPU image lacks it) ---
import importlib.util, subprocess, sys
if importlib.util.find_spec("torch_xla") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.9.0", "torch_xla[tpu]==2.9.0",
                    "-f", "https://storage.googleapis.com/libtpu-releases/index.html"], check=True)
print(subprocess.run([sys.executable, "-c", "import torch, torch_xla; print('torch', torch.__version__, '| torch_xla', torch_xla.__version__)"],
                     capture_output=True, text=True).stdout)
'''

BENCH_PY = r'''
import os, statistics, sys, time
sys.path.insert(0, "/kaggle/working/vishalam_src")
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.utils.checkpoint as xla_ckpt
import vishalam.model.model as vm
from vishalam.model.config import ModelConfig
from vishalam.model.model import Vishalam
from vishalam.train.muon import Muon

SIZES = {
    "125m": dict(vocab_size=50304, d_model=768, n_layers=12, n_heads=12, ffn_dim=2048, max_seq_len=4096),
    "350m": dict(vocab_size=50304, d_model=1024, n_layers=24, n_heads=16, ffn_dim=2816, max_seq_len=4096),
}


def use_flash_attention():
    import torch.nn.functional as TF
    from torch_xla.experimental.custom_kernel import flash_attention

    class _F:
        def __getattr__(self, n):
            return getattr(TF, n)

        def scaled_dot_product_attention(self, q, k, v, attn_mask=None, is_causal=False, **kw):
            assert attn_mask is None
            return flash_attention(q, k, v, causal=is_causal, sm_scale=q.shape[-1] ** -0.5)

    vm.F = _F()


def run(index, size, micro, ckpt, flash, steps=12, warm=4, seq=4096):
    vm.checkpoint = lambda fn, *a, use_reentrant=False: xla_ckpt.checkpoint(fn, *a)
    if flash:
        use_flash_attention()
    dev = torch_xla.device()
    world, rank = xr.world_size(), xr.global_ordinal()
    torch.manual_seed(0)
    model = Vishalam(ModelConfig(**SIZES[size])).to(dev)
    model.grad_checkpointing = ckpt
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.dim() == 2]
    other = [p for n, p in model.named_parameters() if not (n.startswith("blocks.") and p.dim() == 2)]
    opts = [Muon(hidden, lr=0.02), torch.optim.AdamW(other, lr=3e-4, betas=(0.9, 0.95), foreach=True)]
    params = [p for p in model.parameters()]
    g = torch.Generator().manual_seed(rank)
    x = torch.randint(0, 50304, (micro, seq + 1), generator=g).to(dev)
    times = []
    for i in range(steps):
        t = time.time()
        with torch.autocast("xla", dtype=torch.bfloat16):
            _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
        xm.all_reduce(xm.REDUCE_SUM, [p.grad for p in params], scale=1.0 / world)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=False)
        xm.mark_step()
        xm.wait_device_ops()
        times.append(time.time() - t)
        if rank == 0:
            print(f"  step {i}: {times[-1]:.2f}s loss {loss.item():.3f}", flush=True)
    if rank == 0:
        dt = statistics.median(times[warm:])
        tps = micro * seq * world / dt
        print(f"RESULT {size} chips={world} micro={micro}/chip ckpt={ckpt} flash={flash}: {tps:,.0f} tokens/s "
              f"({dt:.2f}s/step, first step {times[0]:.0f}s) -> 10.5B tokens = {10.5e9 / tps / 3600:,.0f} h, "
              f"6B = {6e9 / tps / 3600:,.0f} h", flush=True)


if __name__ == "__main__":
    size, micro, ckpt, flash = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1"
    torch_xla.launch(run, args=(size, micro, ckpt, flash))
'''

RUN = '''\
# --- 3. Run each config in its own process (an out-of-memory config only loses itself) ---
import subprocess, sys
open("/kaggle/working/bench.py", "w").write(BENCH_PY)
CONFIGS = [  # size, micro per chip, activation checkpointing, flash attention
    ("350m", 4, 1, 0),
    ("350m", 4, 1, 1),
    ("350m", 8, 1, 1),
    ("350m", 4, 0, 1),
]
summary = []
for size, micro, ckpt, flash in CONFIGS:
    print(f"==== {size} micro={micro} ckpt={ckpt} flash={flash}", flush=True)
    try:
        r = subprocess.run([sys.executable, "/kaggle/working/bench.py", size, str(micro), str(ckpt), str(flash)],
                           capture_output=True, text=True, timeout=1500)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = f"TIMEOUT {e}"
    lines = [l for l in out.splitlines() if "RESULT" in l or "step " in l or "Error" in l or "error" in l]
    print("\\n".join(lines[-20:]) or out[-3000:], flush=True)
    summary += [l for l in out.splitlines() if l.startswith("RESULT")] or [f"FAILED {size} micro={micro} ckpt={ckpt} flash={flash}"]
print("\\n==== SUMMARY ====")
print("\\n".join(summary))
'''


def main() -> None:
    write = ["# --- 2. Vishalam model source (embedded) + benchmark script ---", "import os"]
    for f in FILES:
        src = (ROOT / f).read_text(encoding="utf-8")
        write.append(f"os.makedirs(os.path.dirname('/kaggle/working/vishalam_src/{f}'), exist_ok=True)")
        write.append(f"open('/kaggle/working/vishalam_src/{f}', 'w').write({src!r})")
    write.append(f"BENCH_PY = {BENCH_PY!r}")
    write.append('print("source written")')
    cells = [INSTALL, "\n".join(write) + "\n", RUN]
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "vishalam_tpu_bench"
    out.mkdir(exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    meta = {"id": f"{user}/vishalam-tpu-bench", "title": "vishalam tpu bench", "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_tpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": [], "machine_shape": "TpuV5E8"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
