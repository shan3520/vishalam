"""Score a model as a plain code completer on HumanEval: no agent, tools or repair attempts.

    python -m vishalam.eval.base_completion --weights run/model.pt --tokenizer tokenizer.json \
        --out results/base-350m.json --workers 4

One greedy completion per task (pass@1): the prompt is fed after <|endoftext|>, which is what the start
of a new file looks like in the pretraining data, and generation stops at the usual top-level markers.
This shows what the pretrained model can do on its own, to compare with the agent's HumanEval score.
With --samples K --temperature T, a task gets up to K sampled attempts and pass@k counts tasks with a
correct one among the first k. The hidden tests pick the winner, so this is an upper bound on what
retrying can gain: a real agent can only choose using the tests or examples it can see.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import tempfile
import time
import zlib
from pathlib import Path

STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n@")


def truncate(text: str) -> tuple[str, bool]:
    """The completion up to the first top-level stop marker, and whether one was found."""
    cuts = [i for i in (text.find(s) for s in STOPS) if i >= 0]
    return (text[: min(cuts)], True) if cuts else (text, False)


def complete(model, cfg, device, tok, prompt: str, max_new: int = 320, chunk: int = 16, temperature: float = 0.0,
             generator=None) -> str:
    from vishalam.infer.generate import Session

    ids = [tok.eot_id] + tok.encode_ordinary(prompt)
    session = Session(model, cfg, device, max_len=min(cfg.max_seq_len, len(ids) + max_new + 1))
    session.feed(ids)
    out: list[int] = []
    while len(out) < max_new and session.remaining > 0:
        new = session.sample({tok.eot_id}, max_new=chunk, temperature=temperature, generator=generator)
        if not new:
            break
        out += new
        ended = new[-1] == tok.eot_id
        text, stopped = truncate(tok.decode([i for i in out if i != tok.eot_id]))
        if stopped or ended:
            return text
    return truncate(tok.decode(out))[0]


def score(task: dict, completion: str, sandbox) -> tuple[bool, str]:
    from vishalam.eval.harness import score_hidden

    root = Path(tempfile.mkdtemp(prefix="vish_base_"))
    try:
        (root / "solution.py").write_text(task["prompt"] + completion + "\n", encoding="utf-8")
        return score_hidden(root, task, sandbox)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def load_tasks() -> list[dict]:
    from datasets import load_dataset

    from vishalam.eval.harness import load_humaneval

    prompts = {r["task_id"]: r["prompt"] for r in load_dataset("openai/openai_humaneval", split="test")}
    return [{**t, "prompt": prompts[t["id"]]} for t in load_humaneval()]


def summarize(records: list[dict], ks: tuple[int, ...] = (1,)) -> dict:
    """`first_pass` is the 1-based attempt that first passed (None: none did)."""
    clean = [r for r in records if not r["family_overlap"]]

    def rate(rs: list[dict], k: int) -> float:
        return sum(r["first_pass"] is not None and r["first_pass"] <= k for r in rs) / len(rs) if rs else 0.0

    table: dict = {"n": len(records), "n_clean": len(clean)}
    for k in ks:
        table[f"pass_at_{k}"] = rate(records, k)
        table[f"pass_at_{k}_clean"] = rate(clean, k)
    return table


_STATE = None


def _init(weights: str, tokenizer: str, samples: int, temperature: float) -> None:
    global _STATE
    import torch

    from vishalam.agent.sandbox import SubprocessSandbox
    from vishalam.infer.generate import load_model, load_tokenizer

    torch.set_num_threads(1)  # one core per process: decoding one token at a time gains nothing from threads
    model, cfg, device = load_model(weights, "cpu")
    _STATE = (model, cfg, device, load_tokenizer(tokenizer), SubprocessSandbox(timeout=15), samples, temperature)


def _work(task: dict) -> dict:
    import torch

    model, cfg, device, tok, sandbox, samples, temperature = _STATE
    t0 = time.time()
    attempts, first_pass = [], None
    for i in range(samples):
        gen = torch.Generator().manual_seed(zlib.crc32(task["id"].encode()) + i)
        completion = complete(model, cfg, device, tok, task["prompt"], temperature=temperature, generator=gen)
        passed, out = score(task, completion, sandbox)
        attempts.append({"completion": completion, "passed": passed, "output": out[-300:]})
        if passed:  # later attempts cannot change pass@k
            first_pass = i + 1
            break
    return {"id": task["id"], "first_pass": first_pass, "family_overlap": task["family_overlap"],
            "seconds": round(time.time() - t0, 1), "attempts": attempts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1, help="attempts per task")
    ap.add_argument("--temperature", type=float, default=0.0, help="0: greedy; use > 0 with several samples")
    args = ap.parse_args()
    tasks = load_tasks()[: args.limit]
    records: list[dict] = []
    ks = tuple(k for k in (1, 2, 5, 10, 20) if k <= args.samples)
    init = (args.weights, args.tokenizer, args.samples, args.temperature)
    with mp.get_context("spawn").Pool(args.workers, _init, init) as pool:
        for i, rec in enumerate(pool.imap(_work, tasks)):
            records.append(rec)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"{i + 1}/{len(tasks)} so far {summarize(records, ks)}", flush=True)
    table = summarize(records, ks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"weights": args.weights, "samples": args.samples,
                                          "temperature": args.temperature, "table": table, "records": records},
                                         indent=1))
    print(json.dumps(table), flush=True)


if __name__ == "__main__":
    main()
