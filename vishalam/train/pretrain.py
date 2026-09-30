"""Vishalam pretraining loop.

Single GPU / CPU:
    python -m vishalam.train.pretrain --config configs/vishalam-30m.yaml --data-dir data/shards --out runs/30m
Kaggle (2x T4):
    torchrun --nproc_per_node=2 -m vishalam.train.pretrain --config configs/vishalam-350m.yaml \
        --data-repo shantanu3520/vishalam-data --ckpt-repo shantanu3520/vishalam --out /kaggle/working/run \
        --max-hours 11.3

Resumes automatically from <out>/ckpt.pt, else from the HF checkpoint repo. Stops cleanly after
--max-hours (saving + uploading first) so Kaggle/Colab session limits never lose work.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vishalam.model.config import ModelConfig, load_yaml
from vishalam.model.model import Vishalam
from vishalam.train.ckpt import load_checkpoint, save_checkpoint, save_weights, unwrap
from vishalam.train.data import GlobalSampler, ShardStore, TrainLoader, merge_local_manifests, val_batches


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, device_type: str):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {"fused": True} if device_type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, **kwargs)


def build_optimizers(model: torch.nn.Module, tcfg: dict, device_type: str) -> list[torch.optim.Optimizer]:
    """Default: one AdamW (unchanged behaviour). With `optimizer: muon`: Muon for the transformer's
    hidden 2-D matrices, AdamW for the embedding/tied head and the 1-D norm weights.
    Every param group gets `base_lr`; the schedule multiplies it by a factor each step."""
    if tcfg.get("optimizer", "adamw") != "muon":
        opts = [build_optimizer(model, tcfg["lr"], tcfg["weight_decay"], device_type)]
    else:
        from vishalam.train.muon import Muon

        qkv, hidden, adam_decay, adam_no_decay = [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("blocks.") and p.dim() == 2:
                (qkv if name.endswith("attn.qkv.weight") else hidden).append(p)
            elif p.dim() >= 2:
                adam_decay.append(p)       # embedding (tied with the output head)
            else:
                adam_no_decay.append(p)    # RMSNorm weights
        muon = Muon([{"params": hidden}, {"params": qkv, "split": 3}], lr=tcfg["muon_lr"],
                    momentum=tcfg.get("muon_momentum", 0.95), weight_decay=tcfg.get("muon_weight_decay", 0.0))
        kwargs = {"fused": True} if device_type == "cuda" else {}
        adam = torch.optim.AdamW([{"params": adam_decay, "weight_decay": tcfg["weight_decay"]},
                                  {"params": adam_no_decay, "weight_decay": 0.0}],
                                 lr=tcfg["lr"], betas=(0.9, 0.95), eps=1e-8, **kwargs)
        opts = [muon, adam]
    for opt in opts:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data/shards", help="local shard dir (cache dir if --data-repo)")
    ap.add_argument("--data-repo", default=None, help="HF dataset repo holding the shards")
    ap.add_argument("--data-dirs", default=None,
                    help="comma list of local shard dirs, each with a manifest.json (Kaggle kernel outputs)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (vocab size check)")
    ap.add_argument("--resume-from", default=None,
                    help="dir with ckpt.pt + log.csv from a previous session (e.g. the other ping-pong kernel)")
    ap.add_argument("--ckpt-repo", default=None, help="HF model repo for checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--max-steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--stop-at-step", type=int, default=None, help="pause (save) at this step (tests)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true", help="ignore a recent heartbeat from another session")
    args = ap.parse_args()

    # ---- distributed setup ----
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world = 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    master = rank == 0
    torch.manual_seed(args.seed + rank)

    raw = load_yaml(args.config)
    name, tcfg = raw["name"], raw["train"]
    mcfg = ModelConfig(**raw["model"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- precision: bf16 where supported (A100/L4...), fp16 + loss scaling on T4 ----
    if device_type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext
    )

    # ---- model / optimizer ----
    model = Vishalam(mcfg).to(device)
    model.grad_checkpointing = bool(tcfg.get("grad_checkpointing", False))
    if master:
        print(f"{name}: {unwrap(model).num_params() / 1e6:.1f}M params, device={device}x{world}, amp={amp_dtype}", flush=True)
    optimizers = build_optimizers(model, tcfg, device_type)
    if master:
        print("optimizers:", " + ".join(f"{type(o).__name__}({sum(p.numel() for g in o.param_groups for p in g['params']) / 1e6:.1f}M)" for o in optimizers), flush=True)

    # ---- data ----
    if args.data_dirs:
        store = ShardStore(out / "_shard_cache")  # shards are read in place (absolute paths)
        manifest = merge_local_manifests([d for d in args.data_dirs.split(",") if d])
    else:
        store = ShardStore(args.data_dir, args.data_repo)
        manifest = store.manifest()
    seq_len = tcfg["seq_len"]
    sampler = GlobalSampler(store, manifest["train"], seq_len, seed=args.seed)
    global_batch = tcfg["global_batch_seqs"]
    loader = TrainLoader(sampler, global_batch, tcfg["micro_batch"], rank, world)
    total_steps = args.max_steps or int(tcfg["total_tokens"] // (global_batch * seq_len))
    vbatches = val_batches(store, manifest["val"], seq_len, tcfg["micro_batch"], tcfg["eval_batches"],
                           rank, world, device) if manifest.get("val") else []

    # the shards must have been made with a tokenizer whose vocab matches the model
    try:
        from vishalam.tokenizer.bpe import Tokenizer
        tok_vocab = Tokenizer.load(args.tokenizer or store.path("tokenizer.json")).vocab_size
        assert tok_vocab == mcfg.vocab_size, f"tokenizer vocab {tok_vocab} != model vocab {mcfg.vocab_size}"
    except FileNotFoundError:
        pass

    # ---- hub + resume ----
    hub = None
    if args.ckpt_repo:
        from vishalam.train.hub import HubSync
        # every rank reads the heartbeat, so all ranks agree on whether to abort
        hb = HubSync(args.ckpt_repo).pull(f"{name}/latest/heartbeat.json", out / f"_hb{rank}")
        if hb is not None and not args.force:
            beat = json.loads(hb.read_text())
            age_min = (time.time() - beat["utc"]) / 60
            if beat.get("running") and age_min < tcfg["save_every_minutes"] + 20:
                raise SystemExit(
                    f"'{name}' looks like it is already training on {beat.get('host')} "
                    f"(heartbeat {age_min:.0f} min ago). Two sessions would overwrite each other's "
                    f"checkpoints. Stop the other session, wait, or pass --force if it is dead.")
        if master:
            hub = HubSync(args.ckpt_repo)
    ckpt_path = out / "ckpt.pt"
    if not ckpt_path.exists() and args.ckpt_repo:
        if master:
            hub.pull(f"{name}/latest/ckpt.pt", out / "_hub")
            prev_log = hub.pull(f"{name}/latest/log.csv", out / "_hub")
            if prev_log is not None:
                shutil.copy(prev_log, out / "log.csv")  # keep one continuous loss curve
        if ddp:
            dist.barrier()
        pulled = out / "_hub" / name / "latest" / "ckpt.pt"
        if pulled.exists():
            ckpt_path = pulled
    if not ckpt_path.exists() and args.resume_from:
        prev = Path(args.resume_from)
        if (prev / "ckpt.pt").exists():
            ckpt_path = prev / "ckpt.pt"  # read in place (Kaggle inputs are read-only)
            if master and (prev / "log.csv").exists() and not (out / "log.csv").exists():
                shutil.copy(prev / "log.csv", out / "log.csv")  # one continuous loss curve
    step = 0
    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path, model, optimizers, scaler, map_location=device)
        step = state["step"]
        del state
        if master:
            print(f"resumed from {ckpt_path} at step {step}", flush=True)
    if ddp:
        dist.barrier()  # every rank has loaded before the downloaded copy is removed
    if master:
        shutil.rmtree(out / "_hub", ignore_errors=True)  # keep /kaggle/working small
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    eager_model = model  # DDP (or plain) module: used for eval, no_sync and as the compile fallback
    compiled = False
    if tcfg.get("compile") and device_type == "cuda":
        try:
            model = torch.compile(model)
            compiled = True
        except Exception as e:  # e.g. no Triton on this platform
            if master:
                print(f"torch.compile unavailable ({type(e).__name__}: {e}); training eagerly", flush=True)

    # ---- logging ----
    log_path = out / "log.csv"
    if master and not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens", "loss", "val_loss", "lr", "grad_norm", "tok_per_s", "scale"])

    def log(row):
        if master:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    def heartbeat(running: bool) -> Path:
        p = out / "heartbeat.json"
        p.write_text(json.dumps({"utc": time.time(), "host": platform.node(), "step": step, "running": running}))
        return p

    if hub:
        hub.push(heartbeat(True), f"{name}/latest/heartbeat.json", f"{name} session start")

    def save(final: bool = False, running: bool = True):
        if not master:
            return
        save_checkpoint(out / "ckpt.pt", model, optimizers, scaler,
                        step, {"name": name, "tokens": step * global_batch * seq_len})
        files = [(out / "ckpt.pt", f"{name}/latest/ckpt.pt"), (log_path, f"{name}/latest/log.csv"),
                 (heartbeat(running), f"{name}/latest/heartbeat.json")]
        if final:
            save_weights(out / "model.pt", model, mcfg.to_dict())
            files.append((out / "model.pt", f"{name}/final/model.pt"))
        if hub:
            hub.push_async(files, f"{name} step {step}", squash=True)
        print(f"saved checkpoint at step {step}", flush=True)

    @torch.no_grad()
    def evaluate() -> float:
        eager_model.eval()
        losses = []
        for x, y in vbatches:
            with autocast():
                _, loss = eager_model(x, y)
            losses.append(loss.float())
        eager_model.train()
        v = torch.stack(losses).mean() if losses else torch.tensor(float("nan"), device=device)
        if ddp:
            dist.all_reduce(v, op=dist.ReduceOp.AVG)
        return v.item()

    # ---- train ----
    t_start = time.time()
    last_save = time.time()
    model.train()
    while step < total_steps:
        t0 = time.time()
        factor = lr_at(step, total_steps, 1.0, tcfg["warmup_steps"], tcfg["min_lr_ratio"])
        lr = factor * tcfg["lr"]  # logged value: the AdamW LR (Muon groups scale their own base_lr)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * factor
        batches = loader.step_batches(step, device)

        def forward_backward(m) -> torch.Tensor:
            total = torch.zeros((), device=device)
            for i, (x, y) in enumerate(batches):
                sync = contextlib.nullcontext() if (not ddp or i == len(batches) - 1) else eager_model.no_sync()
                with sync:
                    with autocast():
                        _, loss = m(x, y, z_loss=tcfg.get("z_loss", 0.0))
                    loss = loss / len(batches)
                    scaler.scale(loss).backward()
                total += loss.detach()
            return total

        try:
            loss_sum = forward_backward(model)
        except Exception as e:
            if not compiled:
                raise
            # compiled graphs failed on this machine: drop back to eager and redo this step
            if master:
                print(f"torch.compile failed ({type(e).__name__}: {str(e)[:200]}); falling back to eager", flush=True)
            model, compiled = eager_model, False
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss_sum = forward_backward(model)
        for opt in optimizers:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        for opt in optimizers:
            scaler.step(opt)  # skipped automatically if grads are inf/nan
        scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        step += 1

        if ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        dt = time.time() - t0
        tok_s = global_batch * seq_len / dt
        val = ""
        if vbatches and (step % tcfg["eval_every"] == 0 or step == total_steps):
            val = f"{evaluate():.4f}"
        if master and (step % 10 == 0 or step == 1 or val):
            mem = f" mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device_type == "cuda" else ""
            print(f"step {step}/{total_steps} loss {loss_sum.item():.4f} {('val ' + val) if val else ''} "
                  f"lr {lr:.2e} gnorm {grad_norm.item():.2f} {tok_s:,.0f} tok/s{mem}", flush=True)
            log([step, step * global_batch * seq_len, f"{loss_sum.item():.4f}", val, f"{lr:.3e}",
                 f"{grad_norm.item():.3f}", f"{tok_s:.0f}", scaler.get_scale() if scaler.is_enabled() else 1])

        elapsed_h = (time.time() - t_start) / 3600
        # stop decision must be identical on all ranks
        pause = elapsed_h > args.max_hours or (args.stop_at_step is not None and step >= args.stop_at_step)
        stop = torch.tensor([1.0 if pause else 0.0], device=device)
        if ddp:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item() > 0:
            if master:
                print(f"time budget reached ({elapsed_h:.2f} h) — saving and exiting", flush=True)
            break
        if (time.time() - last_save) / 60 > tcfg["save_every_minutes"]:
            save()
            last_save = time.time()

    save(final=step >= total_steps, running=False)
    if hub:
        hub.wait()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if master:
        print("finished" if step >= total_steps else f"paused at step {step}/{total_steps}", flush=True)


if __name__ == "__main__":
    main()
