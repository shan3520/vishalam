"""Train the Vishalam byte-level BPE from scratch.

Algorithm: pre-tokenize the corpus into chunks, count unique chunks, then repeatedly merge the
most frequent adjacent pair. Pair counts are updated incrementally (only words containing the
merged pair are touched) and the best pair is found with a lazy max-heap, so training on a
~1 GB sample is practical in pure Python.

Usage:
    python -m vishalam.tokenizer.bpe_train --input sample.txt [--input more.txt] \
        --vocab-size 50304 --out tokenizer.json
"""
from __future__ import annotations

import argparse
import heapq
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Iterable

import regex

from .bpe import NUM_SPECIAL_SLOTS, PRETOKEN_PATTERN, Tokenizer

_PAT = regex.compile(PRETOKEN_PATTERN)


def _count_chunks(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(_PAT.findall(t))
    return c


def count_chunks(texts: Iterable[str], workers: int = 1, batch: int = 2000) -> Counter:
    """Count pre-token chunks over an iterable of documents (optionally in parallel)."""
    total: Counter = Counter()
    buf: list[str] = []

    def batches():
        nonlocal buf
        for t in texts:
            buf.append(t)
            if len(buf) >= batch:
                yield buf
                buf = []
        if buf:
            yield buf

    if workers <= 1:
        for b in batches():
            total.update(_count_chunks(b))
    else:
        with Pool(workers) as pool:
            for c in pool.imap_unordered(_count_chunks, batches()):
                total.update(c)
    return total


def train_bpe(
    chunk_counts: Counter,
    vocab_size: int,
    min_frequency: int = 2,
    verbose: bool = True,
) -> list[tuple[int, int]]:
    n_merges = vocab_size - 256 - NUM_SPECIAL_SLOTS
    assert n_merges > 0, "vocab_size too small"

    words: list[list[int]] = []
    freqs: list[int] = []
    for chunk, f in chunk_counts.items():
        if f < min_frequency:
            continue
        b = chunk.encode("utf-8")
        if len(b) < 2:
            continue  # single bytes never contribute pairs
        words.append(list(b))
        freqs.append(f)

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            where[p].add(wi)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    t0 = time.time()
    while len(merges) < n_merges and heap:
        neg, pair = heapq.heappop(heap)
        cur = pair_counts.get(pair, 0)
        if cur <= 0:
            continue
        if -neg != cur:  # stale heap entry
            heapq.heappush(heap, (-cur, pair))
            continue
        new_id = 256 + len(merges)
        merges.append(pair)
        a, b = pair
        changed: set[tuple[int, int]] = set()
        for wi in where.pop(pair, ()):
            w = words[wi]
            if len(w) < 2:
                continue
            f = freqs[wi]
            # quick skip if pair no longer present
            found = False
            for i in range(len(w) - 1):
                if w[i] == a and w[i + 1] == b:
                    found = True
                    break
            if not found:
                continue
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                changed.add(p)
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for p in zip(out, out[1:]):
                pair_counts[p] += f
                where[p].add(wi)
                changed.add(p)
        pair_counts.pop(pair, None)
        for p in changed:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_counts:
                del pair_counts[p]
        if verbose and len(merges) % 2000 == 0:
            print(f"  {len(merges)}/{n_merges} merges  ({time.time() - t0:.0f}s)", flush=True)
    if len(merges) < n_merges and verbose:
        print(f"  corpus exhausted after {len(merges)} merges (wanted {n_merges})")
    return merges


def _iter_files(paths: list[str], doc_sep: str | None):
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            if doc_sep is None:
                # one document per ~1 MB block, split on line boundaries
                buf: list[str] = []
                size = 0
                for line in f:
                    buf.append(line)
                    size += len(line)
                    if size > 1_000_000:
                        yield "".join(buf)
                        buf, size = [], 0
                if buf:
                    yield "".join(buf)
            else:
                for doc in f.read().split(doc_sep):
                    yield doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="UTF-8 text file (repeatable)")
    ap.add_argument("--vocab-size", type=int, default=50304)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    counts = count_chunks(_iter_files(args.input, None), workers=args.workers)
    print(f"{len(counts):,} unique chunks counted in {time.time() - t0:.0f}s", flush=True)
    merges = train_bpe(counts, args.vocab_size, args.min_frequency)
    tok = Tokenizer(merges)
    tok.save(args.out)
    print(f"saved {args.out}: vocab_size={tok.vocab_size} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
