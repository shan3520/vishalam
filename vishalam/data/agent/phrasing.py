"""Sentence frames around task descriptions. TEST frames never appear in training data."""
from __future__ import annotations

import random

TRAIN_FRAMES = [
    "Write {d}.", "Implement {d}.", "Create {d}.", "Please write {d}.", "I need {d}.",
    "Can you write {d}?", "Define {d}.", "Write me {d}.", "Build {d} in Python.",
    "Code up {d}.", "Please implement {d}.", "Add {d}.", "Make {d}.",
]
TEST_FRAMES = ["Could you implement {d}?", "I'd like {d}.", "Help me write {d}.", "Your task: write {d}."]

FIX_FRAMES_TRAIN = [
    "There is a bug in `{m}.py`. It should contain {d}. Find and fix the bug.",
    "`{m}.py` is broken. It is supposed to define {d}. Fix it.",
    "Fix the bug in `{m}.py`, which should provide {d}.",
    "The code in `{m}.py` has a mistake. It should implement {d}. Please fix it.",
]
FIX_FRAMES_TEST = ["Something is wrong in `{m}.py`; it should define {d}. Can you repair it?"]


def make_wrap(split: str):
    frames = TRAIN_FRAMES if split == "train" else TEST_FRAMES

    def wrap(desc: str, rng: random.Random) -> str:
        return rng.choice(frames).format(d=desc)

    return wrap


def fix_request(desc: str, module: str, split: str, rng: random.Random) -> str:
    frames = FIX_FRAMES_TRAIN if split == "train" else FIX_FRAMES_TEST
    return rng.choice(frames).format(d=desc, m=module)


# ---- worked examples in requests (input -> expected output), the way people usually ask ----
def _call(fname: str, args) -> str:
    return f"{fname}({', '.join(repr(a) for a in args)})"


def _ex_sentence(calls):
    parts = [f"`{c}` should return `{e}`" for c, e in calls]
    return "For example, " + (", and ".join([", ".join(parts[:-1]), parts[-1]]) if len(parts) > 1 else parts[0]) + "."


def _ex_arrows(calls):
    return "Examples:\n" + "\n".join(f"- `{c}` -> `{e}`" for c, e in calls)


def _ex_doctest(calls):
    return "Like this:\n" + "\n".join(f">>> {c}\n{e}" for c, e in calls)


def _ex_equals(calls):
    return "; ".join(f"`{c} == {e}`" for c, e in calls) + "."


def _ex_gives(calls):  # held out: test requests only
    return "Here is how it should work: " + ", ".join(f"`{c}` gives `{e}`" for c, e in calls) + "."


TRAIN_EXAMPLE_STYLES = [_ex_sentence, _ex_arrows, _ex_doctest, _ex_equals]
TEST_EXAMPLE_STYLES = [_ex_gives]


def examples_text(fname: str, cases, split: str, rng: random.Random) -> str:
    """Worked examples for a request; `cases` are (args, expected) pairs. The agent's tests use exactly
    these, so it never has to invent expected values."""
    calls = [(_call(fname, args), repr(exp)) for args, exp in cases]
    styles = TRAIN_EXAMPLE_STYLES if split == "train" else TEST_EXAMPLE_STYLES
    return rng.choice(styles)(calls)


def with_examples(request: str, examples: str) -> str:
    sep = "\n" if "\n" in examples else " "
    return request.rstrip() + sep + examples
