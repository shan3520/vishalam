"""Agent fine-tuning (SFT) on episodes, starting from a pretrained Vishalam model.

    python -m vishalam.train.sft --config configs/sft-m1-125m.yaml --episodes data/m1/train_mixed.jsonl \
        --tokenizer tokenizer.json --out runs/sft
Kaggle (2x T4): prefix with `torchrun --standalone --nproc_per_node=2`, add --ckpt-repo to sync to HF.

Episodes are packed back-to-back into seq_len windows. The loss only counts tokens the model has to
produce (tool calls, done message, end-of-text); requests and tool results are context only.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from vishalam.agent.format import Step, encode_episode
from vishalam.model.config import ModelConfig, load_yaml
from vishalam.model.model import Vishalam
from vishalam.tokenizer.bpe import Tokenizer
from vishalam.train.ckpt import load_checkpoint, save_checkpoint, save_weights
from vishalam.train.pretrain import build_optimizer, lr_at


def tokenize_episodes(paths: list[str], tok: Tokenizer, max_len: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ids_list, mask_list, dropped = [], [], 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                steps = [Step.from_dict(s) for s in ep["steps"]]
                ids, is_model = encode_episode(tok, ep["request"], steps)
                if len(ids) > max_len:
                    dropped += 1
                    continue
                ids_list.append(np.asarray(ids, dtype=np.int32))
                mask_list.append(np.asarray(is_model, dtype=np.uint8))
    if dropped:
        print(f"dropped {dropped} episodes longer than {max_len} tokens", flush=True)
    return ids_list, mask_list


class PackedEpochs:
    """Deterministic packing per epoch: shuffle episodes (seed+epoch), concatenate, cut windows."""

    def __init__(self, ids: list[np.ndarray], masks: list[np.ndarray], seq_len: int, seed: int):
        self.ids, self.masks, self.seq_len, self.seed = ids, masks, seq_len, seed
        self._epoch, self._data = None, None
        total = sum(len(x) for x in ids)
        self.windows_per_epoch = (total - 1) // seq_len

    def _build(self, epoch: int):
        order = np.random.default_rng(self.seed + epoch).permutation(len(self.ids))
        ids = np.concatenate([self.ids[i] for i in order])
        mask = np.concatenate([self.masks[i] for i in order])
        self._epoch, self._data = epoch, (ids, mask)

    def window(self, g: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        epoch, w = divmod(g, self.windows_per_epoch)
        if epoch != self._epoch:
            self._build(epoch)
        ids, mask = self._data
        s = w * self.seq_len
        chunk = ids[s : s + self.seq_len + 1].astype(np.int64)
        m = mask[s + 1 : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:], m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episodes", action="append", required=True, help="episode JSONL (repeatable)")
    ap.add_argument("--val-episodes", default=None)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--init", default=None, help="pretrained weights (model.pt); overrides config")
    ap.add_argument("--ckpt-repo", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world, local_rank = 0, 1, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    raw = load_yaml(args.config)
    name, tcfg = raw["name"], raw["train"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.load(args.tokenizer)

    # ---- model from pretrained weights ----
    init = args.init or raw.get("init")
    hub = None
    if args.ckpt_repo and master:
        from vishalam.train.hub import HubSync
        hub = HubSync(args.ckpt_repo)
    if init and init.startswith("hub:"):  # e.g. hub:vishalam-125m/final/model.pt
        from vishalam.train.hub import HubSync
        if master:
            HubSync(args.ckpt_repo).pull(init[4:], out / "_init")
        if ddp:
            dist.barrier()
        init = str(out / "_init" / init[4:])
    if init:
        state = torch.load(init, map_location="cpu", weights_only=False)
        mcfg = ModelConfig(**state["config"])
        model = Vishalam(mcfg)
        model.load_state_dict({k: v.float() for k, v in state["model"].items()})
    else:
        mcfg = ModelConfig(**raw["model"])
        model = Vishalam(mcfg)  # from scratch (tests / ablations only)
    assert mcfg.vocab_size == tok.vocab_size, "tokenizer/model vocab mismatch"
    model.to(device)
    model.grad_checkpointing = bool(tcfg.get("grad_checkpointing", False))

    seq_len = tcfg["seq_len"]
    ids, masks = tokenize_episodes(args.episodes, tok, seq_len)
    data = PackedEpochs(ids, masks, seq_len, args.seed)
    global_batch, micro = tcfg["global_batch_seqs"], tcfg["micro_batch"]
    assert global_batch % (micro * world) == 0
    accum = global_batch // (micro * world)
    total_steps = args.max_steps or math.ceil(tcfg["epochs"] * data.windows_per_epoch / global_batch)
    if master:
        n_tok = sum(len(x) for x in ids)
        print(f"{name}: {len(ids)} episodes, {n_tok / 1e6:.1f}M tokens, {data.windows_per_epoch} windows/epoch, "
              f"{total_steps} steps", flush=True)

    val = None
    if args.val_episodes:
        vids, vmasks = tokenize_episodes([args.val_episodes], tok, seq_len)
        val = PackedEpochs(vids, vmasks, seq_len, 0)

    if device.startswith("cuda"):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext
    optimizer = build_optimizer(model, tcfg["lr"], tcfg["weight_decay"], "cuda" if device.startswith("cuda") else "cpu")

    ckpt_path = out / "ckpt.pt"
    if not ckpt_path.exists() and args.ckpt_repo:
        if master:
            (hub or __import__("vishalam.train.hub", fromlist=["HubSync"]).HubSync(args.ckpt_repo)).pull(
                f"{name}/latest/ckpt.pt", out / "_hub")
        if ddp:
            dist.barrier()
        if (out / "_hub" / name / "latest" / "ckpt.pt").exists():
            ckpt_path = out / "_hub" / name / "latest" / "ckpt.pt"
    step = 0
    if ckpt_path.exists():
        step = load_checkpoint(ckpt_path, model, optimizer, scaler, map_location=device)["step"]
        if master:
            print(f"resumed at step {step}", flush=True)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    log_path = out / "log.csv"
    if master and not log_path.exists():
        log_path.write_text("step,loss,val_loss,lr,grad_norm,tok_per_s\n")

    def batch(g0: int, source: PackedEpochs):
        xs, ys, ms = zip(*(source.window(g) for g in range(g0, g0 + micro)))
        to = lambda a, dt: torch.from_numpy(np.stack(a)).to(device=device, dtype=dt)
        return to(xs, torch.long), to(ys, torch.long), to(ms, torch.float32)

    @torch.no_grad()
    def evaluate() -> float:
        if val is None:
            return float("nan")
        model.eval()
        losses = []
        n = min(val.windows_per_epoch, 16 * micro)
        for g in range(rank * micro, n, micro * world):
            x, y, m = batch(g, val)
            with autocast():
                _, loss = model(x, y, loss_mask=m)
            losses.append(loss.float())
        model.train()
        v = torch.stack(losses).mean() if losses else torch.tensor(float("nan"), device=device)
        if ddp:
            dist.all_reduce(v, op=dist.ReduceOp.AVG)
        return v.item()

    def save(final=False):
        if not master:
            return
        save_checkpoint(out / "ckpt.pt", model, optimizer, scaler, step, {"name": name})
        files = [(out / "ckpt.pt", f"{name}/latest/ckpt.pt"), (log_path, f"{name}/latest/log.csv")]
        if final:
            save_weights(out / "model.pt", model, mcfg.to_dict())
            files.append((out / "model.pt", f"{name}/final/model.pt"))
        if hub:
            hub.push_async(files, f"{name} step {step}")

    t_start, last_save = time.time(), time.time()
    model.train()
    while step < total_steps:
        t0 = time.time()
        lr = lr_at(step, total_steps, tcfg["lr"], tcfg["warmup_steps"], tcfg["min_lr_ratio"])
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        base = step * global_batch + rank * micro * accum
        for i in range(accum):
            x, y, m = batch(base + i * micro, data)
            sync = contextlib.nullcontext() if (not ddp or i == accum - 1) else model.no_sync()
            with sync:
                with autocast():
                    _, loss = model(x, y, loss_mask=m, z_loss=tcfg.get("z_loss", 0.0))
                scaler.scale(loss / accum).backward()
            loss_sum += loss.detach() / accum
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        vl = evaluate() if (val is not None and (step % tcfg["eval_every"] == 0 or step == total_steps)) else None
        if master and (step % 10 == 0 or step == 1 or vl is not None):
            tps = global_batch * seq_len / (time.time() - t0)
            print(f"step {step}/{total_steps} loss {loss_sum.item():.4f}"
                  f"{f' val {vl:.4f}' if vl is not None else ''} lr {lr:.2e} {tps:,.0f} tok/s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{loss_sum.item():.4f},{'' if vl is None else f'{vl:.4f}'},{lr:.3e},{gnorm.item():.3f},{tps:.0f}\n")
        stop = torch.tensor([1.0 if (time.time() - t_start) / 3600 > args.max_hours else 0.0], device=device)
        if ddp:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            break
        if (time.time() - last_save) / 60 > tcfg.get("save_every_minutes", 30):
            save()
            last_save = time.time()
    save(final=step >= total_steps)
    if hub:
        hub.wait()
    if ddp:
        dist.destroy_process_group()
    if master:
        print("finished" if step >= total_steps else f"paused at {step}/{total_steps}", flush=True)


if __name__ == "__main__":
    main()
