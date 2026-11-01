"""Verify generated tasks (vishalam.data.agent.gen_tasks) and turn them into agent episodes; mix with a
capped share of the template-family episodes.

    python -m vishalam.data.agent.gen_episodes --cands c0.jsonl,c1.jsonl --families m1/train_mixed.jsonl \
        --out m1/train_mixed.jsonl --workers 4

A candidate becomes a task only if, run in the sandbox:
  * the reference solution runs on at least 6 of its inputs and gives literal values (expected outputs
    come from running it, as for the families),
  * the independently written second solution gives the same output on every one of those inputs,
  * the outputs vary (not all equal; a yes/no function returns both answers),
  * it imports only small standard-library modules and is at most 40 lines,
and it passes decontamination: no name or 8-word phrase shared with HumanEval / MBPP-test prompts, and
nothing resembling a held-out family (names and concept keywords), so test_unseen stays unseen.
Each task gives a "write" episode (12% start with a wrong approach - another task's code - and
rewrite the file), and 30% also a "fix" episode. Family episodes are capped per family (default 40) so the
templates no longer dominate.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from vishalam.data.agent.families.base import Case, TaskInstance, normalize_source

ALLOWED_IMPORTS = {"math", "itertools", "collections", "functools", "string", "re", "heapq", "bisect",
                   "statistics", "datetime", "operator", "typing", "fractions", "decimal"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "breakpoint"}
# concepts of the held-out families (vishalam.data.agent.families.HELD_OUT): a generated task that
# mentions one is dropped, so the unseen-family score keeps measuring generalisation
HELD_OUT_KEYWORDS = re.compile(r"\b(" + "|".join([
    r"peaks?\b", r"alternat\w* (sum|sign)", r"every (nth|n-th|kth|k-th)", r"longest increasing",
    r"increasing run", r"interleav", r"(start|begin)\w* with a vowel", r"mask", r"sort words",
    r"digital root", r"perfect numbers?", r"nearest multiple", r"round\w* to a multiple", r"invert",
    r"swap\w* (the )?keys and values", r"inverse mapping", r"anagram", r"letter grades?", r"renam",
    r"spiral", r"rotat", r"border", r"majority", r"overlapping intervals?", r"merg\w* (the )?intervals?",
    r"leaders?\b", r"sliding window", r"window max", r"roman", r"ipv4", r"ip address", r"pig latin",
    r"wrap", r"catalan", r"happy", r"(to|in) base\b", r"radix", r"base conversion", r"prime factor",
    r"factori[sz]", r"dense rank", r"custom order", r"given order", r"median", r"power set",
    r"(all|every) subsets?", r"hanoi", r"coins?\b", r"make change", r"valid date", r"leap year",
    r"versions?\b", r"duration strings?", r"parse (a |the )?duration",
]) + ")")
RUNNER = '''
import ast, copy, json, sys
sys.setrecursionlimit(5000)
def load(path):
    ns = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    return ns.get(FNAME)
fa = load("a.py")
try:
    fb = load("b.py")
except Exception:
    fb = None
rows = []
for args in ast.literal_eval(open("inputs.txt", encoding="utf-8").read()):
    row = []
    for f in (fa, fb):
        try:
            row.append(repr(f(*copy.deepcopy(args))) if f else None)
        except Exception:
            row.append(None)
    rows.append(row)
print("@@" + json.dumps(rows))
'''


def shingles(text: str, n: int = 8) -> set[str]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def benchmark_blocklist() -> tuple[set[str], set[str]]:
    """Function names and 8-word phrases of HumanEval and MBPP-test (needs the datasets)."""
    from datasets import load_dataset

    names, grams = set(), set()
    for r in load_dataset("openai/openai_humaneval", split="test"):
        names.add(r["entry_point"].lower())
        grams |= shingles(r["prompt"])
    for r in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        m = re.search(r"def\s+(\w+)\s*\(", r["code"])
        if m:
            names.add(m.group(1).lower())
        grams |= shingles(r["text"])
    return names, grams


def held_out_names() -> set[str]:
    from vishalam.data.agent.families import UNSEEN_FAMILIES

    out = set()
    for fam in UNSEEN_FAMILIES:
        for n in fam.names:
            out |= {n.format(**v).lower() for v in fam.variants} if "{" in n else {n.lower()}
    return out


def static_check(c: dict) -> tuple[list[str], list[tuple]] | None:
    """Parse the reference; returns (params, normalized input tuples) or None if unusable."""
    try:
        tree = ast.parse(c["solution"])
        inputs = ast.literal_eval(c["inputs_src"])
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    if len(c["solution"].splitlines()) > 40 or not isinstance(inputs, (list, tuple)):
        return None
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == c["fname"]]
    if not fn:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] not in ALLOWED_IMPORTS for a in node.names):
            return None
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            return None
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return None
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return None
    a = fn[0].args
    if a.vararg or a.kwarg or a.kwonlyargs:
        return None
    params = [x.arg for x in a.args]
    lo, hi = len(params) - len(a.defaults), len(params)
    norm = []
    for e in inputs:
        if hi == 1 and not (isinstance(e, tuple) and len(e) == 1):
            e = (e,)
        if isinstance(e, list) and lo <= len(e) <= hi and hi > 1:
            e = tuple(e)
        if isinstance(e, tuple) and lo <= len(e) <= hi and len(repr(e)) <= 300:
            norm.append(e)
    return (params, norm) if len(norm) >= 6 else None


def run_both(c: dict, inputs: list[tuple], sandbox) -> list[list] | None:
    root = Path(tempfile.mkdtemp(prefix="vish_gen_"))
    try:
        (root / "a.py").write_text(c["solution"], encoding="utf-8")
        (root / "b.py").write_text(c.get("solution2") or "", encoding="utf-8")
        (root / "inputs.txt").write_text(repr(inputs), encoding="utf-8")
        out = sandbox.run_python(root, RUNNER.replace("FNAME", repr(c["fname"]))).output
        m = re.search(r"@@(\[.*\])\s*$", out, re.S)
        return json.loads(m.group(1)) if m else None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def verify(c: dict, sandbox) -> tuple[TaskInstance, list[Case]] | None:
    st = static_check(c)
    if st is None:
        return None
    params, inputs = st
    rows = run_both(c, inputs, sandbox)
    if rows is None or len(rows) != len(inputs):
        return None
    cases = []
    for args, (ra, rb) in zip(inputs, rows):
        if ra is None:
            continue
        if rb != ra:
            return None  # the two solutions disagree: the request is ambiguous or one is wrong
        try:
            val = ast.literal_eval(ra)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
        if repr(val) != ra or len(ra) > 300:
            return None
        cases.append(Case(args, val))
    if len(cases) < 6:
        return None
    outs = {repr(x.expected) for x in cases}
    if len(outs) < (2 if all(isinstance(x.expected, bool) for x in cases) else 3):
        return None
    module = c["fname"] if c["fname"] not in sys.stdlib_module_names else f"{c['fname']}_impl"
    return TaskInstance("generated", c.get("topic", "generated"), c["fname"], params, {}, c["request"], module,
                        normalize_source(c["solution"]), [], [], request_source="ai-generated",
                        description=c["request"]), cases


def split_cases(cases: list[Case], rng: random.Random) -> tuple[list[Case], list[Case]]:
    """2-3 typical cases with different answers (+ a small one as an edge case) for the agent's own tests,
    the rest hidden - as families.base.make_instance does."""
    by_size = sorted(cases, key=lambda c: len(repr(c.args)))
    small, rest = by_size[0], by_size[1:]
    typical = rest[len(rest) // 4: max(len(rest) // 4 + 1, len(rest) * 3 // 4)] or rest
    rng.shuffle(typical)
    agent, seen = [], set()
    for c in typical:
        if repr(c.expected) not in seen:
            agent.append(c)
            seen.add(repr(c.expected))
        if len(agent) == rng.choice([2, 3]):
            break
    if len(agent) < 2:
        agent = typical[:2]
    agent.append(small)
    hidden = [c for c in cases if c not in agent][:12]
    return agent, hidden


def rename_function(source: str, old: str, new: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == old:
            node.name = new
        elif isinstance(node, ast.Name) and node.id == old:
            node.id = new  # recursive calls
    return ast.unparse(tree) + "\n"


_SB = None


def _sandbox():
    global _SB
    if _SB is None:
        from vishalam.agent.sandbox import SubprocessSandbox
        _SB = SubprocessSandbox(timeout=5)
    return _SB


def _verify_work(c):
    try:
        r = verify(c, _sandbox())
    except Exception:
        return None
    if r is None:
        return None
    task, cases = r
    return task.__dict__ | {"cases": [[repr(x.args), repr(x.expected)] for x in cases]}


def _episode_work(item):
    from vishalam.data.agent.trajectories import synthesize

    d, seed, other = item
    rng = random.Random(seed)
    cases = [Case(ast.literal_eval(a), ast.literal_eval(e)) for a, e in d.pop("cases")]
    agent, hidden = split_cases(cases, rng)
    task = TaskInstance(**{**d, "agent_cases": agent, "hidden_cases": hidden})
    eps = []
    wrong = rename_function(other["ref_source"], other["fname"], task.fname) if other and rng.random() < 0.12 else None
    for _ in range(6):
        ep = synthesize(task, rng, "create", _sandbox(), "train", wrong_first=wrong)
        if ep is not None:
            eps.append(ep)
            break
        wrong = None
    if rng.random() < 0.3:
        for _ in range(4):
            ep = synthesize(task, rng, "fix", _sandbox(), "train")
            if ep is not None:
                eps.append(ep)
                break
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="comma list of gen_tasks outputs")
    ap.add_argument("--families", required=True, help="template-family episodes (train_mixed.jsonl)")
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--per-name", type=int, default=40, help="tasks sharing a function name (names follow the theme)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-benchmarks", action="store_true", help="skip HumanEval/MBPP decontamination (tests)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    stats: Counter = Counter()

    cands = [json.loads(l) for p in args.cands.split(",") for l in open(p, encoding="utf-8") if l.strip()]
    stats["candidates"] = len(cands)
    bench_names, bench_grams = (set(), set()) if args.no_benchmarks else benchmark_blocklist()
    blocked_names = held_out_names() | bench_names
    keep, per_name, seen_req = [], Counter(), set()
    for c in cands:
        text = (c["fname"] + " " + c["request"]).lower().replace("_", " ")
        if c["fname"].lower() in blocked_names or HELD_OUT_KEYWORDS.search(text):
            stats["dropped: held-out concept or benchmark name"] += 1
            continue
        if shingles(c["request"]) & bench_grams:
            stats["dropped: benchmark phrase"] += 1
            continue
        key = " ".join(re.findall(r"[a-z0-9]+", c["request"].lower()))
        if key in seen_req or per_name[c["fname"]] >= args.per_name:
            stats["dropped: duplicate"] += 1
            continue
        seen_req.add(key)
        per_name[c["fname"]] += 1
        keep.append(c)
    with Pool(args.workers) as pool:
        tasks = [t for t in pool.imap(_verify_work, keep, chunksize=16) if t is not None]
        stats["dropped: failed verification"] = len(keep) - len(tasks)
        stats["tasks"] = len(tasks)
        print(json.dumps(dict(stats)), flush=True)
        items = [(t, args.seed * 1_000_003 + i, rng.choice(tasks) if len(tasks) > 1 else None)
                 for i, t in enumerate(tasks)]
        generated = []
        for k, eps in enumerate(pool.imap(_episode_work, items, chunksize=8)):
            generated += eps
            if (k + 1) % 2000 == 0:
                print(f"episodes from {k + 1}/{len(items)} tasks: {len(generated)}", flush=True)

    fam = defaultdict(list)
    for l in open(args.families, encoding="utf-8"):
        if l.strip():
            ep = json.loads(l)
            fam[ep["family_id"]].append(ep)
    family_eps = []
    for fid in sorted(fam):
        rng.shuffle(fam[fid])
        family_eps += fam[fid][: args.per_family]
    allep = generated + family_eps
    rng.shuffle(allep)
    with open(args.out, "w", encoding="utf-8") as f:
        for ep in allep:
            f.write(json.dumps(ep) + "\n")
    rewrites = sum(sum(s["call"]["name"] == "write_file" and s["call"]["args"]["path"] == f"{e['module']}.py"
                       for s in e["steps"]) > 1 for e in generated)
    stats.update({"generated episodes": len(generated), "family episodes": len(family_eps),
                  "rewrite episodes": rewrites})
    Path(args.out).with_name("gen_stats.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)), flush=True)


if __name__ == "__main__":
    main()
