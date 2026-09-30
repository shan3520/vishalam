"""Expert iteration (self-improvement): the agent attempts many fresh tasks; attempts whose final code
passes the hidden tests become new training episodes. Everything comes from Vishalam's own system:
our task families, our model, our sandbox. No outside AI.

    python -m vishalam.train.expert_iter --weights model.pt --tokenizer tokenizer.json \
        --n-tasks 3000 --samples 4 --temperature 0.7 --out data/m1/self_round1.jsonl

Filters: the episode must end with `done`, have no invalid tool calls, pass the hidden tests, and
(at most one kept episode per task) prefer the shortest successful attempt.
"""
from __future__ import annotations

import argparse
import json
import random
import time

from vishalam.agent.runtime import Agent
from vishalam.agent.sandbox import SubprocessSandbox
from vishalam.data.agent.families import TRAIN_FAMILIES, make_instance
from vishalam.data.agent.phrasing import make_wrap
from vishalam.data.agent.trajectories import cases_to_json, eval_task
from vishalam.eval.harness import run_task
from vishalam.infer.generate import load_model, load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n-tasks", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, cfg, device = load_model(args.weights)
    tok = load_tokenizer(args.tokenizer)
    sb = SubprocessSandbox(timeout=5)
    rng = random.Random(args.seed)
    wrap = make_wrap("train")
    t0 = time.time()
    kept = tried = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ti in range(args.n_tasks):
            if (time.time() - t0) / 3600 > args.max_hours:
                break
            task = make_instance(rng.choice(TRAIN_FAMILIES), rng, "train", wrap=wrap)
            kind = "fix" if rng.random() < 0.2 else "create"
            t = eval_task(task, rng, kind, "train")
            if t is None:
                continue
            best = None
            for s in range(args.samples):
                agent = Agent(model, cfg, device, tok, temperature=args.temperature, seed=args.seed + ti * 97 + s)
                rec = run_task(agent, t, sb)
                tried += 1
                if rec["passed"] and rec["finished"] and rec["parse_errors"] == 0:
                    if best is None or len(rec["steps"]) < len(best["steps"]):
                        best = rec
            if best is not None:
                ep = {**{k: t[k] for k in ("kind", "family_id", "category", "fname", "module", "request",
                                           "initial_files", "ref_source", "hidden_cases")},
                      "split": "train", "request_source": "template", "episode_source": "self-generated",
                      "agent_cases": cases_to_json(task.agent_cases), "steps": best["steps"]}
                f.write(json.dumps(ep) + "\n")
                kept += 1
            if (ti + 1) % 25 == 0:
                print(f"tasks {ti + 1}: kept {kept} ({kept / (ti + 1):.1%} solved), "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
    print(json.dumps({"tasks": ti + 1, "attempts": tried, "kept_episodes": kept}))


if __name__ == "__main__":
    main()
