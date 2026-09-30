"""Evaluation: run the agent on tasks, then score its final code with hidden tests it never saw.

Task record fields:
    id, split, request, initial_files {path: content}
    and either  fname + hidden_cases   (generated / hand-written: Python-literal cases)
    or          hidden_code            (HumanEval / MBPP: raw pytest code with a {module} placeholder)
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import tempfile
from pathlib import Path

from vishalam.agent.tools import Workspace
from vishalam.data.agent.trajectories import cases_from_json, render_hidden_tests

HIDDEN_FILE = "test_zz_hidden.py"


def find_module(root: Path, fname: str) -> str | None:
    """Module (file stem) that defines `fname` at top level, ignoring test files."""
    for p in sorted(root.rglob("*.py")):
        if p.name.startswith("test_"):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        if any(isinstance(n, ast.FunctionDef) and n.name == fname for n in tree.body):
            return p.relative_to(root).with_suffix("").as_posix().replace("/", ".")
    return None


def score_hidden(ws_root: Path, task: dict, sandbox) -> tuple[bool, str]:
    fname = task.get("fname")
    module = find_module(ws_root, fname) if fname else None
    if module is None:
        return False, f"no top-level function `{fname}` found"
    tmp = Path(tempfile.mkdtemp(prefix="vish_score_"))
    try:
        for p in ws_root.rglob("*.py"):
            if not p.name.startswith("test_"):  # the agent's own tests are not part of scoring
                dst = tmp / p.relative_to(ws_root)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst)
        if "hidden_code" in task:
            code = task["hidden_code"].replace("{module}", module)
        else:
            code = render_hidden_tests(fname, module, cases_from_json(task["hidden_cases"]))
        (tmp / HIDDEN_FILE).write_text(code, encoding="utf-8")
        out = sandbox.run_tests(tmp, HIDDEN_FILE).output
        ok = " passed" in out and "failed" not in out and "error" not in out.lower()
        return ok, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_task(agent, task: dict, sandbox) -> dict:
    root = Path(tempfile.mkdtemp(prefix="vish_eval_"))
    try:
        ws = Workspace(root, sandbox)
        for name, content in task.get("initial_files", {}).items():
            ws.write_file(name, content)
        res = agent.run(task["request"], ws)
        passed, out = score_hidden(root, task, sandbox)
        return {
            "id": task.get("id"), "split": task["split"], "family_id": task.get("family_id"),
            "kind": task.get("kind", "create"), "passed": passed,
            "first_try": passed and res.test_runs <= 1, "finished": res.finished,
            "test_runs": res.test_runs, "n_steps": len(res.steps), "parse_errors": res.parse_errors,
            "stop_reason": res.stop_reason, "hidden_output": out[-800:],
            "steps": [s.to_dict() for s in res.steps],
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ task sets

def load_generated(path: str | Path) -> list[dict]:
    tasks = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        if line.strip():
            t = json.loads(line)
            t.setdefault("id", f"{t['split']}-{i}")
            tasks.append(t)
    return tasks


def load_handwritten(root: str | Path = "handwritten") -> list[dict]:
    root = Path(root)
    req = root / "requests.jsonl"
    if not req.exists():
        return []
    tasks = []
    for line in open(req, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split") != "test":
            continue
        hidden = root / "hidden" / f"{r['id']}.json"
        if not hidden.exists():
            continue  # hidden tests not written yet
        h = json.loads(hidden.read_text(encoding="utf-8"))
        initial = {r["file"]: r["code"]} if r.get("kind") == "fix" else {}
        tasks.append({"id": r["id"], "split": "handwritten", "kind": r.get("kind", "create"),
                      "request": r["request"], "fname": r["function_name"], "initial_files": initial,
                      "hidden_cases": h["cases"]})
    return tasks


def load_mbpp() -> list[dict]:
    from datasets import load_dataset

    tasks = []
    for r in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        m = re.search(r"def\s+(\w+)\s*\(", r["code"])
        if not m:
            continue
        name = m.group(1)
        body = "\n".join("    " + t for t in r["test_list"])
        setup = r.get("test_setup_code") or ""
        code = f"from {{module}} import *\n{setup}\n\n\ndef test_mbpp():\n{body}\n"
        tasks.append({"id": f"mbpp-{r['task_id']}", "split": "mbpp", "fname": name,
                      "request": f"{r['text'].strip()} The function must be called `{name}`.",
                      "hidden_code": code})
    return tasks


def load_humaneval() -> list[dict]:
    from datasets import load_dataset

    tasks = []
    for r in load_dataset("openai/openai_humaneval", split="test"):
        ep = r["entry_point"]
        code = f"from {{module}} import *\n\n{r['test']}\n\n\ndef test_humaneval():\n    check({ep})\n"
        tasks.append({"id": r["task_id"], "split": "humaneval", "fname": ep,
                      "request": f"Write the Python function `{ep}` described below.\n\n{r['prompt']}",
                      "hidden_code": code})
    return tasks
