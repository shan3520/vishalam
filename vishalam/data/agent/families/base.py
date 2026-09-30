"""Framework for rule-based coding task families.

A Family is a hand-written *kind* of task (e.g. "second largest distinct number"). Each call to
`make_instance` produces a concrete task: a function name, argument names, a variant, an English
request, the reference solution, and test cases whose expected values come from actually running
the reference solution.
"""
from __future__ import annotations

import ast
import copy
import random
import string
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------- family + instance


@dataclass
class Family:
    id: str
    category: str                      # "lists" | "strings" | "math"
    names: list[str]                   # candidate function names
    params: list[list[str]]            # per argument: candidate parameter names
    ref: str                           # reference source; placeholders {f}, {p0}, {p1}, ... + variant keys
    descs: list[str]                   # description templates ("a function `{sig}` that ...")
    gen: Callable[[random.Random, dict], tuple]  # random argument tuple for a variant
    notes: list[list[str]] = field(default_factory=list)  # behaviour sentences; one alternative each
    edge: list[tuple] = field(default_factory=list)       # fixed edge-case argument tuples
    variants: list[dict] = field(default_factory=lambda: [{}])


@dataclass
class Case:
    args: tuple
    expected: object


@dataclass
class TaskInstance:
    family_id: str
    category: str
    fname: str
    params: list[str]
    variant: dict
    request: str
    module: str               # file the solution goes in (without .py)
    ref_source: str           # normalized reference solution (ast.unparse)
    agent_cases: list[Case]   # small visible examples the agent turns into its own tests
    hidden_cases: list[Case]  # used only for scoring
    request_source: str = "template"
    description: str = ""     # the bare description (used to build "fix the bug" requests)
    notes: str = ""           # behaviour sentences appended to requests

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["agent_cases"] = [[list(c.args), c.expected] for c in self.agent_cases]
        d["hidden_cases"] = [[list(c.args), c.expected] for c in self.hidden_cases]
        return d


def normalize_source(src: str) -> str:
    return ast.unparse(ast.parse(src)) + "\n"


def compile_function(source: str, fname: str) -> Callable:
    ns: dict = {}
    exec(compile(source, f"<{fname}>", "exec"), ns)  # trusted: our own reference solutions
    return ns[fname]


def _sig(fname: str, params: list[str]) -> str:
    return f"{fname}({', '.join(params)})"


def _small(case_args: tuple) -> int:
    return len(repr(case_args))


_FORBIDDEN_CACHE: dict[str, set[str]] = {}


def forbidden_names(fam: Family) -> set[str]:
    """Names the reference body uses for itself (locals, loop vars, builtins). A parameter with one of
    these names would shadow it and silently change the reference's behaviour."""
    if fam.id not in _FORBIDDEN_CACHE:
        names: set[str] = set()
        for variant in fam.variants:
            dummy = {"f": "__f__", "sig": "__f__()", **{f"p{i}": f"__p{i}__" for i in range(len(fam.params))}}
            v = {k: (x.format(**dummy) if isinstance(x, str) else x) for k, x in variant.items()}
            tree = ast.parse(fam.ref.format(**dummy, **v))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and not node.id.startswith("__"):
                    names.add(node.id)
                elif isinstance(node, ast.arg) and not node.arg.startswith("__"):
                    names.add(node.arg)
        _FORBIDDEN_CACHE[fam.id] = names
    return _FORBIDDEN_CACHE[fam.id]


def _trivial(case_args: tuple) -> bool:
    """Empty containers / empty strings as the main argument make weak example tests."""
    first = case_args[0] if case_args else None
    return hasattr(first, "__len__") and len(first) <= 1


def make_instance(fam: Family, rng: random.Random, desc_split: str = "train", n_random: int = 40,
                  wrap: Callable[[str, random.Random], str] | None = None) -> TaskInstance:
    variant = rng.choice(fam.variants)
    fname = rng.choice(fam.names).format(**variant)
    params = []
    bad = forbidden_names(fam)
    for choices in fam.params:
        opts = [c for c in choices if c not in params and c != fname and c not in bad]
        if not opts:
            raise ValueError(f"{fam.id}: no safe parameter name among {choices}")
        params.append(rng.choice(opts))
    base = {"f": fname, "sig": _sig(fname, params), **{f"p{i}": p for i, p in enumerate(params)}}
    # variant phrases may mention parameters, e.g. "in `{p0}` but not in `{p1}`"
    variant_fmt = {k: v.format(**base) if isinstance(v, str) else v for k, v in variant.items()}
    subs = {**base, **variant_fmt}

    ref_source = normalize_source(fam.ref.format(**subs))
    fn = compile_function(ref_source, fname)

    # ---- cases from running the reference ----
    seen, cases = set(), []
    candidates = list(fam.edge) + [fam.gen(rng, variant) for _ in range(n_random)]
    for args in candidates:
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        try:
            out = fn(*copy.deepcopy(args))
        except Exception:
            continue
        cases.append(Case(tuple(args), out))
    edge_keys = {repr(e) for e in fam.edge}
    edges = [c for c in cases if repr(c.args) in edge_keys]
    randoms = [c for c in cases if repr(c.args) not in edge_keys]
    # agent's own tests: 2-3 *typical* cases (non-trivial, mid-sized, different answers) + 1 edge case
    typical = sorted((c for c in randoms if not _trivial(c.args)), key=lambda c: _small(c.args))
    typical = typical[len(typical) // 4 : max(len(typical) // 4 + 1, len(typical) * 3 // 4)] or randoms
    rng.shuffle(typical)
    n_typ = rng.choice([2, 3])
    agent, outs = [], set()
    for c in typical:
        if repr(c.expected) not in outs:
            agent.append(c)
            outs.add(repr(c.expected))
        if len(agent) == n_typ:
            break
    for c in typical:  # not enough distinct answers (e.g. boolean results): fill up
        if len(agent) >= n_typ:
            break
        if c not in agent:
            agent.append(c)
    if edges:
        agent.append(rng.choice(edges))
    chosen = {repr(c.args) for c in agent}
    rest = [c for c in randoms if repr(c.args) not in chosen]
    rng.shuffle(rest)
    hidden = [c for c in edges if repr(c.args) not in chosen] + rest
    hidden = hidden[:12]

    # ---- request text ----
    descs = fam.descs
    if len(descs) >= 3:
        descs = descs[:-1] if desc_split == "train" else descs[-1:]
    desc = rng.choice(descs).format(**subs)
    notes = [rng.choice(alts).format(**subs) for alts in fam.notes]
    module = fname
    file_note = ""
    if rng.random() < 0.3:
        module = rng.choice([fname, "utils", "solution", "helpers", f"{fam.category}_utils"])
        file_note = rng.choice([f"Put it in `{module}.py`.", f"Save it as `{module}.py`.",
                                f"The file should be `{module}.py`."])
    body = desc
    request = (wrap or default_wrap)(body, rng)
    if notes:
        request += " " + " ".join(notes)
    if file_note:
        request += " " + file_note
    return TaskInstance(fam.id, fam.category, fname, params, variant, request, module, ref_source,
                        agent, hidden, description=desc, notes=" ".join(notes))


def default_wrap(desc: str, rng: random.Random) -> str:
    return f"Write {desc}."


# ---------------------------------------------------------------- random input helpers
WORDS = ("apple banana cherry dog elephant fish grape house island jungle kite lemon mango night "
         "ocean piano queen river sun tree umbrella violin water xray yellow zebra code python data "
         "list string value number agent test model token window cloud light stone paper bird").split()


def ints(rng, lo=-20, hi=20, n=(0, 8)) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(rng.randint(*n))]


def pos_ints(rng, hi=30, n=(0, 8)) -> list[int]:
    return ints(rng, 0, hi, n)


def word(rng, lo=1, hi=8) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(lo, hi)))


def words(rng, n=(0, 6), real=True) -> list[str]:
    k = rng.randint(*n)
    return [rng.choice(WORDS) if real and rng.random() < 0.7 else word(rng) for _ in range(k)]


def sentence(rng, n=(0, 7), caps=0.2) -> str:
    ws = words(rng, n)
    ws = [w.capitalize() if rng.random() < caps else w for w in ws]
    return " ".join(ws)


def text(rng, n=(0, 20), alphabet=string.ascii_letters + " ,.!?0123456789") -> str:
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(*n)))
