"""Build the M1 agent dataset.

    python -m vishalam.data.agent.build_m1 --out data/m1 --n-train 100000 --workers 4

Big builds are split over machines: the training episodes are 100 fixed chunks (own seed each),
`--part I/N` builds every N-th chunk, and `--merge` joins the parts:
    python -m vishalam.data.agent.build_m1 --out p0 --part 0/5      # part 0 also builds the test sets
    python -m vishalam.data.agent.build_m1 --out m1 --merge p0,p1,p2,p3,p4
A finished chunk is a file (train_parts/chunk_NNNNN.jsonl), so an interrupted part resumes where it
stopped (`--resume-from` copies the chunks of an earlier, interrupted session).

Outputs (JSONL):
    train.jsonl         full episodes (training families, training phrasings)
    test_seen.jsonl     eval tasks: training families, NEW instances, held-out phrasings
    test_unseen.jsonl   eval tasks: families never used in training
    stats.json          counts by kind / family / bug-or-not, for the README
Hand-written seed requests (handwritten/requests.jsonl, split == "seed") are mixed into training
requests; hand-written test requests are never used here.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.data.agent.families import TRAIN_FAMILIES, UNSEEN_FAMILIES, make_instance
from vishalam.data.agent.phrasing import make_wrap
from vishalam.data.agent.trajectories import cases_from_json, eval_task, render_hidden_tests, synthesize

P_FIX = 0.2


def _train_worker(args) -> list[dict]:
    seed, n = args
    rng = random.Random(seed)
    sb = SubprocessSandbox(timeout=5)
    wrap = make_wrap("train")
    out = []
    while len(out) < n:
        fam = rng.choice(TRAIN_FAMILIES)
        task = make_instance(fam, rng, "train", wrap=wrap)
        kind = "fix" if rng.random() < P_FIX else "create"
        ep = synthesize(task, rng, kind, sb, "train")
        if ep is not None:
            out.append(ep)
    return out


def _chunk_worker(item):
    i, work = item
    return i, _train_worker(work)


def _buggy_fails_hidden(t: dict, sb) -> bool:
    """For 'fix' eval tasks, make sure the starting code really fails the hidden tests."""
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="vish_chk_"))
    try:
        for name, content in t["initial_files"].items():
            (root / name).write_text(content, encoding="utf-8")
        (root / "test_hidden.py").write_text(
            render_hidden_tests(t["fname"], t["module"], cases_from_json(t["hidden_cases"])), encoding="utf-8")
        out = sb.run_tests(root).output
        return not (" passed" in out and "failed" not in out and "error" not in out.lower())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _eval_worker(args) -> list[dict]:
    seed, n, unseen = args
    rng = random.Random(seed)
    sb = SubprocessSandbox(timeout=5)
    fams = UNSEEN_FAMILIES if unseen else TRAIN_FAMILIES
    wrap = make_wrap("test")
    split = "test_unseen" if unseen else "test_seen"
    out = []
    while len(out) < n:
        task = make_instance(rng.choice(fams), rng, "test", wrap=wrap)
        kind = "fix" if rng.random() < P_FIX else "create"
        t = eval_task(task, rng, kind, split)
        if t is None or (kind == "fix" and not _buggy_fails_hidden(t, sb)):
            continue
        out.append(t)
    return out


def _split_work(total: int, parts: int, seed: int) -> list[tuple[int, int]]:
    base, extra = divmod(total, parts)
    return [(seed * 1000003 + i, base + (1 if i < extra else 0)) for i in range(parts) if base or i < extra]


def _count(stats: Counter, ep: dict) -> None:
    stats[f"kind:{ep['kind']}"] += 1
    stats[f"family:{ep['family_id']}"] += 1
    stats["with_bug_fix"] += any(s["call"]["name"] == "edit_file" for s in ep["steps"])


def merge(dirs: list[Path], out: Path) -> None:
    """Join the parts' chunk files (each chunk once, in chunk order) and their test sets."""
    chunks: dict[str, Path] = {}
    for d in dirs:
        for f in sorted((d / "train_parts").glob("chunk_*.jsonl")):
            chunks.setdefault(f.name, f)
    stats: Counter = Counter()
    with open(out / "train.jsonl", "w", encoding="utf-8") as fo:
        for name in sorted(chunks):
            for line in chunks[name].read_text(encoding="utf-8").splitlines():
                fo.write(line + "\n")
                _count(stats, json.loads(line))
    for name in ("test_seen.jsonl", "test_unseen.jsonl"):
        src = next((d / name for d in dirs if (d / name).exists()), None)
        if src is None:
            raise SystemExit(f"no part has {name} (part 0 builds the test sets)")
        if src != out / name:
            (out / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
    episodes = sum(v for k, v in stats.items() if k.startswith("kind:"))
    print(f"merged {len(chunks)} chunks, {episodes} episodes", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-test-seen", type=int, default=500)
    ap.add_argument("--n-test-unseen", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=100, help="fixed work units (each has its own seed)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--part", default="0/1", help="I/N: build every N-th chunk starting at I")
    ap.add_argument("--resume-from", default=None, help="an interrupted session's output dir (chunks kept)")
    ap.add_argument("--merge", default=None, help="comma list of part output dirs to join into --out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge([Path(d) for d in args.merge.split(",") if d], out)
        return

    part, nparts = (int(x) for x in args.part.split("/"))
    parts_dir = out / "train_parts"
    parts_dir.mkdir(exist_ok=True)
    if args.resume_from:
        for f in Path(args.resume_from, "train_parts").glob("chunk_*.jsonl"):
            if not (parts_dir / f.name).exists():
                (parts_dir / f.name).write_bytes(f.read_bytes())
    chunk_file = lambda i: parts_dir / f"chunk_{i:05d}.jsonl"  # noqa: E731
    mine = [(i, w) for i, w in enumerate(_split_work(args.n_train, args.chunks, args.seed)) if i % nparts == part]
    todo = [(i, w) for i, w in mine if not chunk_file(i).exists()]
    want = sum(n for _, (_, n) in mine)
    done = want - sum(n for _, (_, n) in todo)
    print(f"part {part}/{nparts}: {len(mine)} chunks, {len(mine) - len(todo)} already done", flush=True)

    with Pool(args.workers) as pool:
        for i, eps in pool.imap_unordered(_chunk_worker, todo):
            tmp = chunk_file(i).with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(ep) + "\n" for ep in eps), encoding="utf-8")
            tmp.replace(chunk_file(i))  # a chunk file is complete or absent
            done += len(eps)
            print(f"train episodes: {done}/{want}", flush=True)
        if part == 0:  # the test sets are built once, by part 0
            for name, n, unseen in [("test_seen", args.n_test_seen, False),
                                    ("test_unseen", args.n_test_unseen, True)]:
                work = [(s + (7 if unseen else 3), k, unseen) for s, k in _split_work(n, args.workers, args.seed)]
                with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                    for tasks in pool.imap_unordered(_eval_worker, work):
                        for t in tasks:
                            f.write(json.dumps(t) + "\n")
                print(f"{name}: {n} tasks", flush=True)
    if nparts == 1:  # a single-machine build is its own merge
        merge([out], out)
    print("done")


if __name__ == "__main__":
    main()
