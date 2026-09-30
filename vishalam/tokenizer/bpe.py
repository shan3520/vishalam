"""Vishalam byte-level BPE tokenizer (written from scratch).

Token id layout:
    0..255                      raw bytes
    256..256+len(merges)-1      learned merges, in merge order (= rank)
    after that                  special tokens (named ones first, then reserved slots)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import regex

# GPT-4-style pre-tokenizer: splits letters, 1-3 digit groups, punctuation runs, newlines,
# and whitespace runs. Runs of indentation spaces stay together, so BPE learns them as tokens.
PRETOKEN_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Named special tokens. The agent protocol (vishalam/agent/format.py) is built on these.
# Never reorder or remove — only append or take from the reserved pool — or trained models break.
NAMED_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|request|>",
    "<|result|>",
    "<|end|>",
    "<|list_files|>",
    "<|read_file|>",
    "<|write_file|>",
    "<|edit_file|>",
    "<|search|>",
    "<|run_tests|>",
    "<|run_python|>",
    "<|done|>",
    "<|path|>",
    "<|content|>",
    "<|old|>",
    "<|new|>",
    "<|pattern|>",
    "<|code|>",
    "<|message|>",
]
NUM_SPECIAL_SLOTS = 128  # named + reserved, so M2/M3 can add tokens without resizing the model


def special_token_list() -> list[str]:
    reserved = [f"<|reserved_{i}|>" for i in range(NUM_SPECIAL_SLOTS - len(NAMED_SPECIAL_TOKENS))]
    return NAMED_SPECIAL_TOKENS + reserved


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], pattern: str = PRETOKEN_PATTERN):
        self.merges = [tuple(m) for m in merges]
        self.pattern = pattern
        self._pat = regex.compile(pattern)
        # rank of a pair == index of the merge; merged token id == 256 + rank
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.token_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        base = 256 + len(self.merges)
        self.special_to_id = {tok: base + i for i, tok in enumerate(special_token_list())}
        self.id_to_special = {i: t for t, i in self.special_to_id.items()}
        self.vocab_size = base + NUM_SPECIAL_SLOTS
        self._special_split = regex.compile(
            "(" + "|".join(regex.escape(t) for t in self.special_to_id) + ")"
        )
        self._encode_chunk = lru_cache(maxsize=1 << 18)(self._encode_chunk_uncached)

    # ---------- special token helpers ----------
    def sid(self, token: str) -> int:
        return self.special_to_id[token]

    @property
    def eot_id(self) -> int:
        return self.special_to_id["<|endoftext|>"]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    # ---------- encoding ----------
    def _encode_chunk_uncached(self, chunk: bytes) -> tuple[int, ...]:
        ids = list(chunk)
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            # merge every occurrence of the best pair in one pass
            pair = (ids[best_i], ids[best_i + 1])
            new_id = 256 + best_rank
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return tuple(ids)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as plain text."""
        out: list[int] = []
        for chunk in self._pat.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text. With allow_special=True, literal '<|...|>' special strings become special ids.

        Use allow_special=False for anything that came from code, files, or tool output, so a
        program printing '<|end|>' can never forge protocol tokens.
        """
        if not allow_special:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_split.split(text):
            if not part:
                continue
            if part in self.special_to_id:
                out.append(self.special_to_id[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    # ---------- decoding ----------
    def decode_bytes(self, ids: list[int]) -> bytes:
        parts = []
        for i in ids:
            if i in self.id_to_special:
                parts.append(self.id_to_special[i].encode("utf-8"))
            else:
                parts.append(self.token_bytes[i])
        return b"".join(parts)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        data = {
            "format": "vishalam-bpe-v1",
            "pattern": self.pattern,
            "merges": [list(m) for m in self.merges],
            "special_tokens": special_token_list(),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["format"] == "vishalam-bpe-v1", data.get("format")
        assert data["special_tokens"] == special_token_list(), "special token list changed"
        return cls([tuple(m) for m in data["merges"]], data["pattern"])
