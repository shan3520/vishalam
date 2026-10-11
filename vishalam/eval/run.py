"""Run evaluations and print the results table.

    python -m vishalam.eval.run --weights runs/sft/model.pt --tokenizer tokenizer.json \
        --sets test_seen,test_unseen,handwritten,humaneval,mbpp --data data/m1 --out results/m1-125m.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from vishalam.agent.runtime import Agent
from vishalam.agent.sandbox import SubprocessSandbox, default_sandbox
from vishalam.eval.harness import load_generated, load_handwritten, load_humaneval, load_mbpp, run_task
from vishalam.infer.generate import load_model, load_tokenizer

M1_BAR = {"test_seen": 0.70, "handwritten": 0.40}


def summarize(records: list[dict]) -> dict:
    by = defaultdict(list)
    for r in records:
        by[r["split"]].append(r)
    table = {}
    for split, rs in by.items():
        n = len(rs)
        table[split] = {
            "n": n,
            "pass": sum(r["passed"] for r in rs) / n,
            "first_try": sum(r["first_try"] for r in rs) / n,
            "finished": sum(r["finished"] for r in rs) / n,
            "avg_steps": sum(r["n_steps"] for r in rs) / n,
            "parse_error_rate": sum(r["parse_errors"] > 0 for r in rs) / n,
            "fix_tasks_pass": (lambda f: sum(r["passed"] for r in f) / len(f) if f else None)(
                [r for r in rs if r["kind"] == "fix"]),
        }
    return table


def print_table(table: dict) -> None:
    print(f"\n{'split':<14}{'n':>6}{'pass (<=3 fixes)':>18}{'first try':>11}{'finished':>10}{'steps':>7}{'parse err':>11}")
    for split, t in table.items():
        print(f"{split:<14}{t['n']:>6}{t['pass']:>17.1%}{t['first_try']:>11.1%}{t['finished']:>10.1%}"
              f"{t['avg_steps']:>7.1f}{t['parse_error_rate']:>11.1%}")
    for split, bar in M1_BAR.items():
        if split in table:
            ok = table[split]["pass"] >= bar
            print(f"M1 bar {split}: {table[split]['pass']:.1%} vs {bar:.0%} -> {'MET' if ok else 'not met'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sets", default="test_seen,test_unseen,handwritten")
    ap.add_argument("--data", default="data/m1")
    ap.add_argument("--limit", type=int, default=None, help="max tasks per set")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sandbox", choices=["auto", "subprocess"], default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-grammar", action="store_true", help="free decoding of tool calls (for comparison)")
    args = ap.parse_args()

    model, cfg, device = load_model(args.weights, args.device)
    tok = load_tokenizer(args.tokenizer)
    agent = Agent(model, cfg, device, tok, grammar=not args.no_grammar)
    sandbox = SubprocessSandbox() if args.sandbox == "subprocess" else default_sandbox()

    loaders = {
        "test_seen": lambda: load_generated(Path(args.data) / "test_seen.jsonl"),
        "test_unseen": lambda: load_generated(Path(args.data) / "test_unseen.jsonl"),
        "handwritten": load_handwritten,
        "humaneval": load_humaneval,
        "mbpp": load_mbpp,
    }
    records = []
    for name in args.sets.split(","):
        tasks = loaders[name]()[: args.limit]
        t0 = time.time()
        for i, task in enumerate(tasks):
            records.append(run_task(agent, task, sandbox))
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                rate = sum(r["passed"] for r in records if r["split"] == task["split"]) / (i + 1)
                print(f"{name}: {i + 1}/{len(tasks)} pass so far {rate:.1%} ({time.time() - t0:.0f}s)", flush=True)
    table = summarize(records)
    print_table(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"weights": args.weights, "table": table, "records": records}, indent=1))


if __name__ == "__main__":
    main()
