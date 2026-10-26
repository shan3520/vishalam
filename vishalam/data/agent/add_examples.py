"""Give existing agent episodes worked examples in their requests (no new rewording needed).

    python -m vishalam.data.agent.add_examples --inp m1/train_mixed.jsonl --out m1/train_ex.jsonl --workers 4

The first M1 data asked for functions without any input/output examples, yet every episode's agent
wrote tests with exact expected values - which a model can only produce by solving the task in its
head. At test time it invented wrong expected values and then "fixed" correct code to match them.
Now (as vishalam.data.agent.trajectories.synthesize does for new data):
  * "fix" requests and P_EXAMPLES of "write" requests list the agent cases as worked examples, which
    are exactly what the episode's tests already check, so the steps stay as they are;
  * the other "write" episodes are re-synthesized with smoke tests (the function runs and returns the
    right kind of value), keeping the (possibly AI-reworded) request text.
"""
from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool

from vishalam.data.agent.families.base import TaskInstance
from vishalam.data.agent.phrasing import examples_text, with_examples
from vishalam.data.agent.trajectories import P_EXAMPLES, cases_from_json, synthesize

_SB = None


def _sandbox():
    global _SB
    if _SB is None:
        from vishalam.agent.sandbox import SubprocessSandbox
        _SB = SubprocessSandbox(timeout=5)
    return _SB


def convert(ep: dict, rng: random.Random, sandbox=None) -> dict:
    if "agent_cases" not in ep:  # not a generated episode (e.g. MBPP): unchanged
        return ep
    cases = cases_from_json(ep["agent_cases"])
    if ep["kind"] == "fix" or rng.random() < P_EXAMPLES:
        ex = examples_text(ep["fname"], [(c.args, c.expected) for c in cases], "train", rng)
        return {**ep, "request": with_examples(ep["request"], ex), "has_examples": True}
    task = TaskInstance(ep["family_id"], ep["category"], ep["fname"], [], {}, ep["request"], ep["module"],
                        ep["ref_source"], cases, cases_from_json(ep["hidden_cases"]),
                        request_source=ep["request_source"], description=ep.get("description", ""),
                        notes=ep.get("notes", ""))
    for _ in range(20):
        new = synthesize(task, rng, "create", sandbox or _sandbox(), "train", examples=False)
        if new is not None:
            return {**ep, **new}  # keeps extra fields such as template_request
    ex = examples_text(ep["fname"], [(c.args, c.expected) for c in cases], "train", rng)
    return {**ep, "request": with_examples(ep["request"], ex), "has_examples": True}


def _work(item):
    i, line, seed = item
    return convert(json.loads(line), random.Random(seed * 1_000_003 + i))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    lines = [l for l in open(args.inp, encoding="utf-8") if l.strip()]
    n_ex = 0
    with Pool(args.workers) as pool, open(args.out, "w", encoding="utf-8") as f:
        for k, ep in enumerate(pool.imap(_work, ((i, l, args.seed) for i, l in enumerate(lines)), chunksize=64)):
            f.write(json.dumps(ep) + "\n")
            n_ex += bool(ep.get("has_examples"))
            if (k + 1) % 5000 == 0:
                print(f"{k + 1}/{len(lines)}", flush=True)
    print(json.dumps({"episodes": len(lines), "with_examples": n_ex}), flush=True)


if __name__ == "__main__":
    main()
