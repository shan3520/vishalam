"""Token-shard data loading with world-size-independent resume.

Shards are raw little-endian uint16 files of token ids (documents separated by <|endoftext|>),
listed in a manifest.json:
    {"tokenizer": "...", "train": [{"name": "train_00000.bin", "tokens": N}, ...],
     "val": [{"name": "val_00000.bin", "tokens": N}]}

Training samples are (seq_len + 1)-token windows. They are numbered globally: sample g lives in
shard s at a chunk position given by a seeded permutation of that shard's chunks. The only
resume state is the global sample counter, so a run can move between Kaggle (2 GPUs) and
Colab (1 GPU) and still see exactly the same data order.
"""
from __future__ import annotations

import json
import threading
import time
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def merge_local_manifests(dirs: list[str | Path]) -> dict:
    """Combine the manifests of several local shard directories (e.g. the outputs of the Kaggle
    data kernels) into one manifest with absolute shard paths. Order: dirs as given, shards in each
    manifest's order, so the global data order is fixed for the whole run."""
    merged: dict = {"train": [], "val": []}
    shas = set()
    for d in dirs:
        d = Path(d)
        m = json.loads((d / "manifest.json").read_text())
        if m.get("tokenizer_sha"):
            shas.add(m["tokenizer_sha"])
        if len(shas) > 1:
            raise ValueError(f"shard dirs were built with different tokenizers: {sorted(shas)}")
        merged["train"] += [{"name": str((d / s["name"]).resolve()), "tokens": s["tokens"]} for s in m["train"]]
        if not merged["val"] and m.get("val"):
            merged["val"] = [{"name": str((d / s["name"]).resolve()), "tokens": s["tokens"]} for s in m["val"]]
    return merged


class ShardStore:
    """Resolves shard names to local files. Optionally downloads from a HF dataset repo,
    prefetches the next shard in the background, and deletes old shards to save disk."""

    def __init__(self, local_dir: str | Path, hf_repo: str | None = None, keep: int = 3):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo = hf_repo
        self.keep = keep
        self._recent: list[str] = []
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Thread] = {}

    def manifest(self) -> dict:
        return json.loads(Path(self.path("manifest.json")).read_text())

    def _download(self, name: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(self.hf_repo, name, repo_type="dataset", local_dir=self.local_dir)
        )

    def path(self, name: str) -> Path:
        p = self.local_dir / name
        with self._lock:
            t = self._inflight.get(name)
        if t is not None:
            t.join()
        if not p.exists():
            if self.hf_repo is None:
                raise FileNotFoundError(p)
            p = self._download(name)
        if name.endswith(".bin"):
            self._touch(name)
        return p

    def prefetch(self, name: str) -> None:
        if self.hf_repo is None or (self.local_dir / name).exists():
            return
        with self._lock:
            if name in self._inflight:
                return
            t = threading.Thread(target=self._download, args=(name,), daemon=True)
            self._inflight[name] = t
        t.start()

    def _touch(self, name: str) -> None:
        if name in self._recent:
            self._recent.remove(name)
        self._recent.append(name)
        if self.hf_repo is None:
            return  # never delete a purely local dataset
        while len(self._recent) > self.keep:
            old = self._recent.pop(0)
            try:
                (self.local_dir / old).unlink()
            except OSError:
                pass


class GlobalSampler:
    def __init__(self, store: ShardStore, shards: list[dict], seq_len: int, seed: int = 1234):
        self.store = store
        self.names = [s["name"] for s in shards]
        self.seq_len = seq_len
        self.seed = seed
        self.chunks = [(s["tokens"] - 1) // seq_len for s in shards]
        self.cum = np.cumsum([0] + self.chunks)
        self.total = int(self.cum[-1])
        assert self.total > 0, "no training chunks"
        self._files: OrderedDict[int, object] = OrderedDict()
        self._perms: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def _file(self, s: int):
        if s not in self._files:
            self._files[s] = open(self.store.path(self.names[s]), "rb", buffering=0)
            if len(self._files) > 2:
                self._files.popitem(last=False)[1].close()
            if s + 1 < len(self.names):
                self.store.prefetch(self.names[s + 1])
        return self._files[s]

    def read(self, s: int, start: int, n: int) -> np.ndarray:
        """Tokens [start, start + n) of shard s, as int64. Plain reads, not np.memmap: on a network-
        mounted input (Kaggle's TPU VM) a failed page-in of a memory map kills the process with SIGBUS,
        while a failed read raises OSError, which is retried with a fresh file handle."""
        for attempt in range(6):
            try:
                f = self._file(s)
                f.seek(2 * start)
                buf = f.read(2 * n)
                if len(buf) != 2 * n:
                    raise OSError(f"short read: {len(buf)} of {2 * n} bytes from {self.names[s]}")
                return np.frombuffer(buf, dtype=np.uint16).astype(np.int64)
            except OSError:
                if attempt == 5:
                    raise
                old = self._files.pop(s, None)
                if old is not None:
                    old.close()
                time.sleep(2 ** attempt)

    def _perm(self, epoch: int, s: int) -> np.ndarray:
        key = (epoch, s)
        if key not in self._perms:
            rng = np.random.default_rng(self.seed + 7919 * s + 104729 * epoch)
            self._perms[key] = rng.permutation(self.chunks[s])
            if len(self._perms) > 2:  # a step spans at most two shards
                self._perms.popitem(last=False)
        return self._perms[key]

    def sample(self, g: int) -> np.ndarray:
        epoch, r = divmod(g, self.total)
        s = bisect_right(self.cum, r) - 1
        c = int(self._perm(epoch, s)[r - self.cum[s]])
        start = c * self.seq_len
        return self.read(s, start, self.seq_len + 1)


class TrainLoader:
    """Yields this rank's micro-batches for a given optimizer step."""

    def __init__(self, sampler: GlobalSampler, global_batch: int, micro_batch: int, rank: int, world: int):
        assert global_batch % (micro_batch * world) == 0, "global batch must divide by micro*world"
        self.sampler = sampler
        self.global_batch = global_batch
        self.micro_batch = micro_batch
        self.rank, self.world = rank, world
        self.accum = global_batch // (micro_batch * world)

    def step_batches(self, step: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        g0 = step * self.global_batch
        mine = [g0 + self.rank + k * self.world for k in range(self.global_batch // self.world)]
        out = []
        for m in range(self.accum):
            idx = mine[m * self.micro_batch : (m + 1) * self.micro_batch]
            arr = torch.from_numpy(np.stack([self.sampler.sample(i) for i in idx]))
            x, y = arr[:, :-1], arr[:, 1:]
            if torch.device(device).type == "cuda":
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            out.append((x, y))
        return out


def val_batches(store: ShardStore, shards: list[dict], seq_len: int, micro_batch: int,
                n_batches: int, rank: int, world: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Fixed, deterministic validation batches (first chunks of the val shards, split by rank)."""
    samp = GlobalSampler(store, shards, seq_len, seed=0)
    total = min(samp.total, n_batches * micro_batch * world)
    idx = list(range(rank, total, world))
    out = []
    for i in range(0, len(idx) - micro_batch + 1, micro_batch):
        # sequential (unpermuted) chunks keep validation identical across runs
        rows = []
        for g in idx[i : i + micro_batch]:
            s = bisect_right(samp.cum, g) - 1
            c = g - samp.cum[s]
            rows.append(samp.read(s, c * seq_len, seq_len + 1))
        arr = torch.from_numpy(np.stack(rows)).to(device)
        out.append((arr[:, :-1], arr[:, 1:]))
    return out
