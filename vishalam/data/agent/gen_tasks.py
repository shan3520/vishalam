"""Generate many *different* coding tasks with Qwen2.5-7B-Instruct (Apache-2.0), for agent fine-tuning.

    python -m vishalam.data.agent.gen_tasks --out cand_0.jsonl --n 30000 --seed 0 --max-hours 10

Why: 100k episodes from 211 hand-written families taught the model those 211 answers; on held-out
families it wrote the solution of the closest family it knew. Here every task is new (OSS-Instruct
style: a random topic x input kind x difficulty steers each one). Per candidate, two generations:
  1. the task: function name, a request (no examples), a reference solution and 8-12 test inputs;
  2. an independent solution written from the request alone.
Nothing is trusted yet: vishalam.data.agent.gen_episodes runs both solutions in a sandbox and keeps a
task only if they agree on every input (and it passes the decontamination filters). Outputs are
candidates (JSONL), tagged so the README can disclose the AI-generated share.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

TOPICS = """inventory counts; shopping cart totals; bank account transactions; temperature readings; exam scores;
sports match results; league tables; voting tallies; word statistics in text; sentence formatting;
file paths; URLs and query strings; email addresses; phone numbers; product codes; serial numbers;
calendar days and weeks; clock times; time durations; timetables; parking fees; taxi fares; tax brackets;
discounts and coupons; currency amounts; recipe ingredient scaling; unit conversion; DNA and RNA strings;
chemical formulas; sensor signals; stock prices; running totals; moving averages; histograms; buckets;
leaderboards; ranking players; tournament brackets; dice games; card hands; board game positions; chess
coordinates; grid paths; mazes on a grid; flood fill on a grid; game of life steps; image pixels as grids;
matrix arithmetic; vectors and points; distances; polygons; rectangles overlap; circles; triangles;
number digits; divisibility; primes; greatest common divisors; fractions; number bases; binary strings;
bit counting; sequences and series; recurrences; combinatorics counts; permutations; subsets;
string compression; run-length encoding; ciphers; palindromes; anagram checks; character frequencies;
bracket matching; simple expression evaluation; tokenizing text; CSV-like lines; key=value config lines;
log lines; HTTP status codes; version tags; semantic tags and labels; nested lists; flattening;
dictionaries of lists; inverting mappings; merging records; grouping records; filtering records;
sorting records by several keys; deduplicating; set operations; intervals and ranges; scheduling meetings;
queues and waiting lines; stacks; undo histories; elevator stops; traffic lights; train schedules;
library loans; school attendance; employee shifts; salary calculations; budgets; loan payments;
interest; population growth; weather summaries; rainfall; elevation profiles; hiking trails;
inventory restocking; warehouse bins; delivery routes; graph adjacency lists; friend networks; family trees;
dependency ordering; reachability; shortest hops; social media hashtags; chat messages; usernames;
password rules; text justification; acronyms; title case; snake and camel case; slugs; number words;
music playlists; song durations; movie ratings; reviews; menus; restaurant orders; seat bookings;
hotel rooms; car mileage; fuel use; step counts; heart rates; calories; sleep logs; plant watering;
pet records; zoo animals; garden plots; chess ratings; puzzle solving; magic squares; sudoku rows;
tic-tac-toe boards; word games; spelling; vowels and consonants; syllables; rhymes; letter shifts""".split(";")
TOPICS = [t.strip() for t in TOPICS if t.strip()]
INPUT_KINDS = ["a list of integers", "a string", "two strings", "a list of strings", "an integer",
               "two integers", "a dict mapping strings to integers", "a list of (string, integer) tuples",
               "a list of lists of integers (a grid)", "a list of dicts", "an integer and a list of integers",
               "a string and an integer", "a list of floats", "a dict mapping strings to lists"]
DIFFICULTY = ["easy", "easy", "medium", "medium", "medium", "harder"]

SYSTEM = "You design small, precise Python programming exercises and solve them correctly."
TASK_PROMPT = """Invent a new, self-contained Python exercise.
Theme: {topic}. Input: {kind}. Difficulty: {level}.
Rules: one function; standard library only; deterministic; no input(), printing, files or randomness;
arguments and the return value are plain Python values (int, float, str, bool, None, list, tuple, dict);
the request must say what type each argument is and exactly what is returned, and settle every edge
case (empty input, ties, missing values), so there is exactly one correct answer. Do not include
examples in the request.
Reply in exactly this format:
FUNCTION: <snake_case name>
REQUEST: <1-4 sentences, written as a user asking for it, mentioning the function as `name(arg1, arg2)` in backticks>
SOLUTION:
```python
<only the function and the imports it needs; no example usage>
```
INPUTS:
```python
[<ONE list literal of 8 to 12 tuples, each holding the arguments of one call in order, e.g. ([3, 1, 2],) for a single list argument or ("abc", 2) for two arguments; literal values only (no comments, variables or expressions like [0] * 5); include edge cases>]
```"""
SOLVE_PROMPT = """{request}

Write the Python function `{signature}`. It is called like this: {calls}
Reply with only one ```python code block containing the function."""

_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def parse_task(text: str) -> dict | None:
    m_f = re.search(r"FUNCTION:\s*`?([A-Za-z_]\w*)`?", text)
    m_r = re.search(r"REQUEST:\s*(.+?)\n\s*SOLUTION:", text, re.S)
    parts = text.split("INPUTS:")
    if not (m_f and m_r and len(parts) == 2):
        return None
    sol, inp = _BLOCK.findall(parts[0]), _BLOCK.findall(parts[1])
    if not (sol and inp):
        return None
    return {"fname": m_f.group(1), "request": " ".join(m_r.group(1).split()), "solution": sol[-1].strip() + "\n",
            "inputs_src": inp[0].strip()}


def parse_inputs(src: str) -> list:
    """The INPUTS block as a list of argument tuples (or bare values). Lenient: one list literal, or one
    literal per line; comments are dropped."""
    lines = [ln.split("  #")[0].split("# ")[0].rstrip().rstrip(",") for ln in src.splitlines()]
    try:
        v = ast.literal_eval("\n".join(lines))
        if isinstance(v, (list, tuple)):
            return list(v)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    out = []
    for ln in lines:
        ln = ln.strip().lstrip("[").rstrip("]") if ln.strip() in ("[", "]") else ln.strip()
        if ln:
            try:
                out.append(ast.literal_eval(ln))
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                pass
    return out


def example_calls(fname: str, inputs: list, k: int = 3) -> str:
    calls = []
    for e in inputs[:k]:
        args = e if isinstance(e, tuple) else (e,)
        calls.append(f"`{fname}({', '.join(repr(a) for a in args)})`")
    return ", ".join(calls)


def signature(source: str, fname: str) -> str | None:
    m = re.search(rf"def\s+{re.escape(fname)}\s*\(([^)]*)\)", source)
    return f"{fname}({m.group(1).strip()})" if m else None


def parse_code(text: str) -> str:
    blocks = _BLOCK.findall(text)
    return (blocks[0] if blocks else text).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--max-hours", type=float, default=10.0)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    t0, rng = time.time(), random.Random(args.seed)
    llm = LLM(MODEL, quantization="awq", dtype="float16", max_model_len=2048, gpu_memory_utilization=0.9,
              seed=args.seed)
    tok = llm.get_tokenizer()
    chat = lambda u: tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                                             tokenize=False, add_generation_prompt=True)  # noqa: E731
    made = kept = 0
    with open(args.out, "a", encoding="utf-8") as f:
        while made < args.n and time.time() - t0 < args.max_hours * 3600:
            specs = [{"topic": rng.choice(TOPICS), "kind": rng.choice(INPUT_KINDS), "level": rng.choice(DIFFICULTY)}
                     for _ in range(min(args.batch, args.n - made))]
            outs = llm.generate([chat(TASK_PROMPT.format(**s)) for s in specs],
                                SamplingParams(temperature=0.9, top_p=0.95, max_tokens=900, seed=rng.randrange(1 << 30)))
            tasks = []
            for s, o in zip(specs, outs):
                t = parse_task(o.outputs[0].text)
                sig = t and signature(t["solution"], t["fname"])
                inputs = parse_inputs(t["inputs_src"]) if sig else []
                if len(inputs) >= 6:
                    tasks.append({**s, **t, "inputs_src": repr(inputs), "signature": sig,
                                  "calls": example_calls(t["fname"], inputs)})
            outs2 = llm.generate([chat(SOLVE_PROMPT.format(**t)) for t in tasks],
                                 SamplingParams(temperature=0.2, top_p=0.95, max_tokens=600, seed=rng.randrange(1 << 30)))
            for t, o in zip(tasks, outs2):
                t["solution2"] = parse_code(o.outputs[0].text)
                t["id"] = f"g{args.seed}-{kept}"
                f.write(json.dumps(t) + "\n")
                kept += 1
            f.flush()
            made += len(specs)
            print(f"candidates {kept}/{made} parsed ({(time.time() - t0) / 3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
