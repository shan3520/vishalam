"""Synthesize full agent episodes by *actually running* each step in a workspace.

Every tool result in the training data is the real output of the real tool, including real
pytest failures from injected bugs, so the model learns to read genuine error messages.
"""
from __future__ import annotations

import ast
import random
import shutil
import tempfile
from pathlib import Path

from vishalam.agent.format import Step, ToolCall
from vishalam.agent.tools import Workspace
from vishalam.data.agent.families.base import Case, TaskInstance
from vishalam.data.agent.mutations import fix_edit, mutate
from vishalam.data.agent.phrasing import fix_request

TEST_NAMES = ["test_basic", "test_example", "test_another_case", "test_more"]


def cases_to_json(cases: list[Case]) -> list[list[str]]:
    """Cases as Python literals (JSON would turn tuples into lists and break equality checks)."""
    return [[repr(tuple(c.args)), repr(c.expected)] for c in cases]


def cases_from_json(rows: list[list[str]]) -> list[Case]:
    return [Case(ast.literal_eval(a), ast.literal_eval(e)) for a, e in rows]


def _has_float(v) -> bool:
    if isinstance(v, float):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_float(x) for x in v)
    return False


def _call_src(fname: str, args: tuple) -> str:
    return f"{fname}({', '.join(repr(a) for a in args)})"


def _assert_line(fname: str, c: Case) -> str:
    exp = f"pytest.approx({c.expected!r})" if _has_float(c.expected) else repr(c.expected)
    if c.expected is None:
        return f"    assert {_call_src(fname, c.args)} is None"
    if isinstance(c.expected, bool):
        return f"    assert {_call_src(fname, c.args)} is {c.expected}"
    return f"    assert {_call_src(fname, c.args)} == {exp}"


def render_tests(fname: str, module: str, cases: list[Case], edge_last: bool = True) -> str:
    uses_approx = any(_has_float(c.expected) for c in cases)
    lines = (["import pytest", ""] if uses_approx else []) + [f"from {module} import {fname}"]
    for i, c in enumerate(cases):
        name = "test_edge_case" if edge_last and i == len(cases) - 1 and len(cases) > 2 else TEST_NAMES[min(i, 3)]
        if name in "\n".join(lines):
            name = f"{name}_{i}"
        lines += ["", "", f"def {name}():", _assert_line(fname, c)]
    return "\n".join(lines) + "\n"


def render_hidden_tests(fname: str, module: str, cases: list[Case]) -> str:
    uses_approx = any(_has_float(c.expected) for c in cases)
    lines = (["import pytest", ""] if uses_approx else []) + [f"from {module} import {fname}"]
    for i, c in enumerate(cases):
        lines += ["", "", f"def test_hidden_{i}():", _assert_line(fname, c)]
    return "\n".join(lines) + "\n"


def _passed(out: str) -> bool:
    return " passed" in out and "failed" not in out and "error" not in out.lower()


DONE_OK = ["Implemented `{f}` in `{file}`. All {n} tests pass.",
           "Done: `{f}` is in `{file}` and all {n} tests pass.",
           "`{f}` is implemented in `{file}`; the {n} tests I wrote pass."]
DONE_FIXED = ["Implemented `{f}` in `{file}`. The first version failed a test, so I fixed it; all {n} tests pass now.",
              "`{f}` is in `{file}`. I fixed a bug found by the tests and all {n} tests pass.",
              "Done. My first attempt at `{f}` had a bug; after the fix all {n} tests in `{test}` pass."]
DONE_REPAIR = ["Fixed the bug in `{file}`; all {n} tests pass now.",
               "The bug in `{f}` is fixed. All {n} tests pass.",
               "Repaired `{f}` in `{file}`, and the {n} tests I added pass."]


def synthesize(task: TaskInstance, rng: random.Random, kind: str, sandbox, split: str = "train",
               p_bug: float = 0.5) -> dict | None:
    """Build one episode. Returns None if the random choices do not give a clean episode (e.g. an
    injected bug that the agent's own tests would not catch); the caller simply tries again."""
    sol_file = f"{task.module}.py"
    test_file = f"test_{task.module}.py"
    tests = render_tests(task.fname, task.module, task.agent_cases)
    n_tests = len(task.agent_cases)
    root = Path(tempfile.mkdtemp(prefix="vish_"))
    try:
        ws = Workspace(root, sandbox)
        steps: list[Step] = []

        def do(name: str, **args) -> str:
            call = ToolCall(name, args)
            out = ws.execute(call)
            steps.append(Step(call, out))
            return out

        initial: dict[str, str] = {}
        request = task.request
        fmt = {"f": task.fname, "file": sol_file, "test": test_file, "n": n_tests}

        if kind == "fix":
            buggy = mutate(task.ref_source, rng)
            if buggy is None:
                return None
            initial = {sol_file: buggy}
            ws.write_file(sol_file, buggy)
            request = fix_request(task.description, task.module, split, rng)
            if task.notes:
                request += " " + task.notes
            do("list_files")
            do("read_file", path=sol_file)
            do("write_file", path=test_file, content=tests)
            if _passed(do("run_tests")):
                return None  # bug not caught by these tests
            old, new = fix_edit(buggy, task.ref_source)
            do("edit_file", path=sol_file, old=old, new=new)
            if not _passed(do("run_tests")):
                return None
            message = rng.choice(DONE_REPAIR).format(**fmt)
        else:
            if rng.random() < 0.4:
                do("list_files")
            do("write_file", path=test_file, content=tests)
            buggy = mutate(task.ref_source, rng) if rng.random() < p_bug else None
            do("write_file", path=sol_file, content=buggy or task.ref_source)
            if rng.random() < 0.1:
                c = rng.choice(task.agent_cases)
                do("run_python", code=f"from {task.module} import {task.fname}\nprint({_call_src(task.fname, c.args)})")
            out = do("run_tests")
            if buggy:
                if _passed(out):
                    return None  # our tests missed this bug: don't teach "done" on buggy code
                if rng.random() < 0.3:
                    do("read_file", path=sol_file)
                old, new = fix_edit(buggy, task.ref_source)
                do("edit_file", path=sol_file, old=old, new=new)
                if not _passed(do("run_tests")):
                    return None
                message = rng.choice(DONE_FIXED).format(**fmt)
            else:
                if not _passed(out):
                    return None
                message = rng.choice(DONE_OK).format(**fmt)
        steps.append(Step(ToolCall("done", {"message": message}), None))

        final = (root / sol_file).read_text(encoding="utf-8")
        assert final == task.ref_source, "episode must end with the reference solution"
        return {
            "kind": kind, "split": split, "family_id": task.family_id, "category": task.category,
            "fname": task.fname, "module": task.module, "request": request,
            "request_source": task.request_source, "description": task.description, "notes": task.notes,
            "initial_files": initial, "steps": [s.to_dict() for s in steps],
            "ref_source": task.ref_source,
            "agent_cases": cases_to_json(task.agent_cases),
            "hidden_cases": cases_to_json(task.hidden_cases),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def eval_task(task: TaskInstance, rng: random.Random, kind: str, split: str) -> dict | None:
    """An evaluation task: just the starting state + request + hidden tests (no solution steps)."""
    initial = {}
    request = task.request
    if kind == "fix":
        buggy = mutate(task.ref_source, rng)
        if buggy is None:
            return None
        initial = {f"{task.module}.py": buggy}
        request = fix_request(task.description, task.module, split, rng)
        if task.notes:
            request += " " + task.notes
    return {
        "kind": kind, "split": split, "family_id": task.family_id, "category": task.category,
        "fname": task.fname, "module": task.module, "request": request,
        "request_source": task.request_source, "initial_files": initial,
        "hidden_cases": cases_to_json(task.hidden_cases),
        "ref_source": task.ref_source,
    }
