"""Load a trained Vishalam checkpoint and generate tokens incrementally (KV cache)."""
from __future__ import annotations

from pathlib import Path

import torch

from vishalam.model.config import ModelConfig
from vishalam.model.model import KVCache, Vishalam
from vishalam.tokenizer.bpe import Tokenizer


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(weights: str | Path, device: str | None = None, quantize_cpu: bool = False):
    """Loads a weights-only export ({"model": state_dict, "config": {...}}) or a full training ckpt
    (needs "config" too). quantize_cpu=True converts Linear layers to int8 on CPU; whether that is
    faster depends on model size, so use fastest_setup() to decide."""
    device = pick_device(device)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["config"])
    model = Vishalam(cfg)
    model.load_state_dict({k: v.float() for k, v in state["model"].items()})
    model.eval()
    if device == "cpu":
        if quantize_cpu:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        model = model.to(device=device, dtype=torch.float16)
    return model, cfg, device


class Session:
    """One growing token sequence with its KV cache: feed() context, sample() continuations."""

    def __init__(self, model, cfg: ModelConfig, device: str, max_len: int | None = None):
        self.model, self.cfg, self.device = model, cfg, device
        self.max_len = max_len or cfg.max_seq_len
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.cache = KVCache(cfg, 1, self.max_len, device, dtype)
        self.ids: list[int] = []
        self._last_logits: torch.Tensor | None = None

    @property
    def remaining(self) -> int:
        return self.max_len - len(self.ids)

    @torch.no_grad()
    def feed(self, ids: list[int], chunk: int = 512) -> None:
        if not ids:
            return
        if len(ids) > self.remaining:
            raise OverflowError("context window full")
        for i in range(0, len(ids), chunk):
            part = torch.tensor([ids[i : i + chunk]], device=self.device)
            logits, _ = self.model(part, cache=self.cache)
            self._last_logits = logits[0, -1].float()
        self.ids.extend(ids)

    @torch.no_grad()
    def sample(self, stop_ids: set[int], max_new: int = 1024, temperature: float = 0.0,
               top_p: float = 0.95, generator: torch.Generator | None = None, grammar=None) -> list[int]:
        """Generate until a stop id (included in the output) or max_new tokens. `grammar` (e.g.
        agent.format.CallGrammar) masks the logits so only grammatical continuations are possible."""
        out: list[int] = []
        for _ in range(max_new):
            if self._last_logits is None or self.remaining <= 0:
                break
            logits = self._last_logits
            if grammar is not None:
                logits = logits + grammar.mask(out, logits)
            if temperature <= 0:
                nxt = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                sorted_p, idx = torch.sort(probs, descending=True)
                keep = torch.cumsum(sorted_p, 0) - sorted_p < top_p
                sorted_p = sorted_p * keep
                choice = torch.multinomial(sorted_p / sorted_p.sum(), 1, generator=generator)
                nxt = int(idx[choice])
            out.append(nxt)
            self.feed([nxt])
            if nxt in stop_ids:
                break
        return out


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.load(path)


def fastest_setup(weights: str | Path, trial_tokens: int = 48) -> tuple[str, bool]:
    """Benchmark the available options on this machine; returns (device, quantize_cpu).
    Small models are overhead-bound (fp32 CPU often wins); larger ones favour GPU or int8."""
    import time

    options = [("cpu", False), ("cpu", True)] + ([("cuda", False)] if torch.cuda.is_available() else [])
    best, best_speed = options[0], 0.0
    for dev, q in options:
        try:
            m, cfg, d = load_model(weights, dev, quantize_cpu=q)
            s = Session(m, cfg, d, max_len=256)
            s.feed(list(range(64)))
            s.sample(set(), max_new=4)  # warm-up
            t = time.time()
            n = len(s.sample(set(), max_new=trial_tokens))
            speed = n / (time.time() - t)
        except Exception:
            continue
        if speed > best_speed:
            best, best_speed = (dev, q), speed
    return best
