"""MBPP training episodes: human-written Python tasks, to make the agent less tied to our task families.

    python -m vishalam.data.agent.mbpp_episodes --out data/mbpp_train.jsonl --repeat 8

MBPP (Austin et al., 2021; CC-BY-4.0, google-research-datasets/mbpp). Only the train, validation and
prompt splits are used; the 500-task test split stays the evaluation set (vishalam.eval.harness).
Each episode is the same shape as a generated one: write the tests, write the solution (MBPP's
reference code), run the tests in the sandbox, done. Tasks whose reference fails its own tests are
skipped. The request wording matches the evaluation's ("... The function must be called `f`.").
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path

from vishalam.agent.format import Step, ToolCall
from vishalam.agent.tools import Workspace
from vishalam.data.agent.trajectories import DONE_OK, _passed

SPLITS = ("train", "validation", "prompt")  # never "test"


def episode_from_record(r: dict, sandbox, rng: random.Random) -> dict | None:
    m = re.search(r"def\s+(\w+)\s*\(", r["code"])
    if not m:
        return None
    fname = m.group(1)
    module = fname if fname not in sys.stdlib_module_names else f"{fname}_impl"
    sol_file, test_file = f"{module}.py", f"test_{module}.py"
    setup = (r.get("test_setup_code") or "").strip()
    body = "\n".join("    " + t for t in r["test_list"])
    tests = f"from {module} import *\n{setup + chr(10) if setup else ''}\n\ndef test_{fname}():\n{body}\n"
    code = r["code"].replace("\r\n", "\n").rstrip() + "\n"
    root = Path(tempfile.mkdtemp(prefix="vish_mbpp_"))
    try:
        ws = Workspace(root, sandbox)
        steps: list[Step] = []

        def do(name: str, **args) -> str:
            call = ToolCall(name, args)
            out = ws.execute(call)
            steps.append(Step(call, out))
            return out

        if rng.random() < 0.4:
            do("list_files")
        do("write_file", path=test_file, content=tests)
        do("write_file", path=sol_file, content=code)
        if not _passed(do("run_tests")):
            return None
        message = rng.choice(DONE_OK).format(f=fname, file=sol_file, test=test_file, n=len(r["test_list"]))
        steps.append(Step(ToolCall("done", {"message": message}), None))
        return {"kind": "create", "split": "train", "family_id": "mbpp", "category": "mbpp",
                "fname": fname, "module": module, "request_source": "mbpp",
                "request": f"{r['text'].strip()} The function must be called `{fname}`.",
                "initial_files": {}, "steps": [s.to_dict() for s in steps], "ref_source": code,
                "mbpp_task_id": r["task_id"]}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> None:
    from datasets import load_dataset

    from vishalam.agent.sandbox import SubprocessSandbox

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=8, help="copies of each task (different incidental steps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng, sb = random.Random(args.seed), SubprocessSandbox(timeout=10)
    records = [r for split in SPLITS for r in load_dataset("google-research-datasets/mbpp", "full", split=split)]
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            for _ in range(args.repeat):
                ep = episode_from_record(r, sb, rng)
                if ep is None:
                    break
                f.write(json.dumps(ep) + "\n")
                kept += 1
    print(f"mbpp episodes: {kept} from {len(records)} tasks x{args.repeat}", flush=True)


if __name__ == "__main__":
    main()
