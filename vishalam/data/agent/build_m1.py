"""Build the M1 agent dataset.

    python -m vishalam.data.agent.build_m1 --out data/m1 --n-train 100000 --workers 4

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-test-seen", type=int, default=500)
    ap.add_argument("--n-test-unseen", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=0, help="work units (default 25 per worker)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks = args.chunks or args.workers * 25

    stats: Counter = Counter()
    with Pool(args.workers) as pool:
        with open(out / "train.jsonl", "w", encoding="utf-8") as f:
            done = 0
            for eps in pool.imap_unordered(_train_worker, _split_work(args.n_train, chunks, args.seed)):
                for ep in eps:
                    f.write(json.dumps(ep) + "\n")
                    stats[f"kind:{ep['kind']}"] += 1
                    stats[f"family:{ep['family_id']}"] += 1
                    stats["with_bug_fix"] += any(s["call"]["name"] == "edit_file" for s in ep["steps"])
                done += len(eps)
                print(f"train episodes: {done}/{args.n_train}", flush=True)
        for name, n, unseen in [("test_seen", args.n_test_seen, False), ("test_unseen", args.n_test_unseen, True)]:
            work = [(s + (7 if unseen else 3), k, unseen) for s, k in _split_work(n, args.workers, args.seed)]
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
                for tasks in pool.imap_unordered(_eval_worker, work):
                    for t in tasks:
                        f.write(json.dumps(t) + "\n")
            print(f"{name}: {n} tasks", flush=True)
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
    print("done")


if __name__ == "__main__":
    main()
