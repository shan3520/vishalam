"""Inject one realistic bug into a correct solution (AST level), and build the edit that fixes it.

Bug kinds: flipped comparisons, swapped arithmetic operators, off-by-one constants / range bounds,
wrong index, and/or swap, removed early-return guard, reversed sort flag, dropped `not`.
"""
from __future__ import annotations

import ast
import difflib
import random

CMP_FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv}
NAME_SWAP = {"min": "max", "max": "min", "set": "list", "any": "all", "all": "any"}
DROP_CALL = {"abs", "reversed", "set", "sorted", "ord", "str"}
ATTR_SWAP = {"lower": "upper", "upper": "lower", "isdigit": "isalpha", "isalpha": "isdigit",
             "startswith": "endswith", "endswith": "startswith", "isupper": "islower", "islower": "isupper",
             "count": "find", "split": "splitlines"}
STR_SWAP = {" ": "", "": " ", "aeiou": "aeio", "_": "-"}


def _sites(tree: ast.AST) -> list[tuple[int, str]]:
    """(node index in ast.walk order, mutation kind) for every applicable site."""
    sites = []
    for i, node in enumerate(ast.walk(tree)):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP_FLIP:
            sites.append((i, "cmp"))
        elif isinstance(node, ast.BinOp) and type(node.op) in BIN_SWAP:
            sites.append((i, "binop"))
        elif isinstance(node, ast.AugAssign) and type(node.op) in BIN_SWAP:
            sites.append((i, "augop"))
        elif isinstance(node, ast.Constant) and type(node.value) is int and not isinstance(node.value, bool):
            sites.append((i, "const"))
        elif isinstance(node, ast.BoolOp):
            sites.append((i, "boolop"))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            sites.append((i, "not"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            sites.append((i, "range"))
        elif isinstance(node, ast.Call) and any(k.arg == "reverse" for k in node.keywords):
            sites.append((i, "reverse"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1 \
                and node.func.id in DROP_CALL:
            sites.append((i, "dropcall"))
        elif isinstance(node, ast.Name) and node.id in NAME_SWAP and isinstance(node.ctx, ast.Load):
            sites.append((i, "name"))
        elif isinstance(node, ast.Attribute) and node.attr in ATTR_SWAP:
            sites.append((i, "attr"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in STR_SWAP:
            sites.append((i, "str"))
        elif isinstance(node, ast.Slice):
            sites.append((i, "slice"))
        elif isinstance(node, ast.FunctionDef):
            if any(isinstance(s, ast.If) and len(s.body) == 1 and isinstance(s.body[0], ast.Return)
                   for s in node.body[:-1]):
                sites.append((i, "guard"))
    return sites


class _Apply(ast.NodeTransformer):
    def __init__(self, target: int, kind: str, rng: random.Random, index_of: dict[int, int]):
        self.target, self.kind, self.rng, self.index_of = target, kind, rng, index_of
        self.done = False

    def visit(self, node):
        if not self.done and self.index_of.get(id(node)) == self.target:
            self.done = True
            return self.mutate(node)
        return super().visit(node)

    def mutate(self, node):
        k, rng = self.kind, self.rng
        if k == "cmp":
            node.ops = [CMP_FLIP[type(node.ops[0])]()]
        elif k in ("binop", "augop"):
            node.op = BIN_SWAP[type(node.op)]()
        elif k == "const":
            node.value = node.value + rng.choice([-1, 1])
        elif k == "boolop":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif k == "not":
            return node.operand
        elif k == "range":
            last = node.args[-1]
            node.args[-1] = ast.BinOp(last, rng.choice([ast.Add(), ast.Sub()]), ast.Constant(1))
        elif k == "reverse":
            for kw in node.keywords:
                if kw.arg == "reverse":
                    kw.value = ast.UnaryOp(ast.Not(), kw.value)
        elif k == "dropcall":
            return node.args[0]
        elif k == "name":
            node.id = NAME_SWAP[node.id]
        elif k == "attr":
            node.attr = ATTR_SWAP[node.attr]
        elif k == "str":
            node.value = STR_SWAP[node.value]
        elif k == "slice":
            if node.lower is None:
                node.lower = ast.Constant(1)
            else:
                node.lower = None
        elif k == "guard":
            idx = [j for j, s in enumerate(node.body[:-1])
                   if isinstance(s, ast.If) and len(s.body) == 1 and isinstance(s.body[0], ast.Return)]
            del node.body[rng.choice(idx)]
        return node


def mutate(source: str, rng: random.Random, tries: int = 8) -> str | None:
    """Return a syntactically valid, different version of `source` with one bug, or None."""
    for _ in range(tries):
        tree = ast.parse(source)
        sites = _sites(tree)
        if not sites:
            return None
        target, kind = rng.choice(sites)
        index_of = {id(n): i for i, n in enumerate(ast.walk(tree))}
        new_tree = _Apply(target, kind, rng, index_of).visit(tree)  # fresh parse each try: mutate in place
        ast.fix_missing_locations(new_tree)
        try:
            out = ast.unparse(new_tree) + "\n"
            compile(out, "<mutant>", "exec")
        except (SyntaxError, ValueError):
            continue
        if out != source:
            return out
    return None


def fix_edit(buggy: str, correct: str) -> tuple[str, str]:
    """Smallest (old, new) pair of line blocks turning `buggy` into `correct`, with enough context
    that `old` occurs exactly once in `buggy`."""
    b, c = buggy.splitlines(), correct.splitlines()
    ops = [op for op in difflib.SequenceMatcher(a=b, b=c, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        raise ValueError("sources are identical")
    i1, i2 = ops[0][1], ops[-1][2]
    j1, j2 = ops[0][3], ops[-1][4]
    if i1 == i2:  # pure insertion: anchor on the previous line
        i1, j1 = i1 - 1, j1 - 1
    while True:
        old = "\n".join(b[i1:i2])
        if old and buggy.count(old) == 1:
            break
        if i1 > 0:
            i1, j1 = i1 - 1, j1 - 1
        elif i2 < len(b):
            i2, j2 = i2 + 1, j2 + 1
        else:
            break
    return "\n".join(b[i1:i2]), "\n".join(c[j1:j2])
