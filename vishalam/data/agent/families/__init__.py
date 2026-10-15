"""Registry of all M1 task families.

HELD_OUT families never appear in training data; they form the `test_unseen_family` split, which
measures whether the agent generalizes to kinds of tasks it has never practised.
"""
from __future__ import annotations

from .base import Case, Family, TaskInstance, make_instance  # noqa: F401
from .dict_tasks import FAMILIES as _DICTS
from .list_tasks import FAMILIES as _LISTS
from .math_tasks import FAMILIES as _MATH
from .matrix_tasks import FAMILIES as _MATRICES
from .more_list_tasks import FAMILIES as _LISTS2
from .more_string_tasks import FAMILIES as _STRINGS2
from .string_tasks import FAMILIES as _STRINGS

ALL: list[Family] = _LISTS + _STRINGS + _MATH + _DICTS + _MATRICES + _LISTS2 + _STRINGS2
BY_ID: dict[str, Family] = {f.id: f for f in ALL}
assert len(BY_ID) == len(ALL), "duplicate family ids"

HELD_OUT = {
    "local_peaks", "alternating_sum", "every_nth", "longest_increasing_run",   # lists
    "interleave", "vowel_start_words", "mask", "sort_words",                   # strings
    "digital_root", "is_perfect", "round_to_multiple",                         # math
}
TRAIN_FAMILIES = [f for f in ALL if f.id not in HELD_OUT]
UNSEEN_FAMILIES = [f for f in ALL if f.id in HELD_OUT]
