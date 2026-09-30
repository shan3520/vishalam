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
