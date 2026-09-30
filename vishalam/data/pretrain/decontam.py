"""Remove training documents that overlap the evaluation benchmarks (HumanEval, MBPP) and the
user's hand-written test requests, using 13-gram overlap on normalized word tokens."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import regex

N = 13
_WORD = regex.compile(r"\w+")


def _ngrams(text: str) -> set[int]:
    words = [w.lower() for w in _WORD.findall(text)]
    out = set()
    for i in range(len(words) - N + 1):
        h = hashlib.blake2b(" ".join(words[i : i + N]).encode(), digest_size=8).digest()
        out.add(int.from_bytes(h, "little"))
    return out


def benchmark_texts() -> list[str]:
    from datasets import load_dataset

    texts = []
    he = load_dataset("openai/openai_humaneval", split="test")
    for r in he:
        texts += [r["prompt"] + r["canonical_solution"], r["test"]]
    for split in ["train", "test", "validation", "prompt"]:
        for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
            texts += [r["text"], r["code"], "\n".join(r["test_list"])]
    hand = Path(__file__).resolve().parents[3] / "handwritten" / "requests.jsonl"
    if hand.exists():
        for line in hand.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line)["request"])
    return texts


class Decontaminator:
    def __init__(self, texts: list[str]):
        self.bad: set[int] = set()
        for t in texts:
            self.bad |= _ngrams(t)

    @classmethod
    def from_benchmarks(cls) -> "Decontaminator":
        return cls(benchmark_texts())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(sorted(self.bad)))

    @classmethod
    def load(cls, path: str | Path) -> "Decontaminator":
        d = cls([])
        d.bad = set(json.loads(Path(path).read_text()))
        return d

    def contaminated(self, text: str) -> bool:
        words = [w.lower() for w in _WORD.findall(text)]
        for i in range(len(words) - N + 1):
            h = hashlib.blake2b(" ".join(words[i : i + N]).encode(), digest_size=8).digest()
            if int.from_bytes(h, "little") in self.bad:
                return True
        return False
